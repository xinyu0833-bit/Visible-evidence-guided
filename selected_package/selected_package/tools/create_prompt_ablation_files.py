# -*- coding: utf-8 -*-
"""
Create prompt-ablation flat prompt JSON files and one test yml per prompt mode.

This script is designed for the existing test_combined_final.py that only accepts:
  python test_combined_final.py -opt xxx.yml

It does not require --prompt_json, --prompt_mode, or --max_images support.
For 10-image quick tests, it creates a symlink dataset with only the first 10 ids
and rewrites the dataset root in the yml.
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List


DEFAULT_MODES = [
    "full_structured",
    "wrong_degradation",
    "random_prompt",
    "generic_prompt",
    "tag_prompt",
    "content_only",
    "degradation_only",
    "content_degradation",
]


def parse_modes(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def sort_ids(ids):
    return sorted(ids, key=lambda x: int(x) if str(x).isdigit() else str(x))


def normalize_id(stem: str) -> str:
    for suffix in ["_masked", "_mask", "-masked", "-mask", ".masked", ".mask"]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def find_file_by_id(folder: Path, image_id: str, kind: str):
    if not folder.exists():
        return None
    exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    suffixes = [""]
    if kind == "masked":
        suffixes = ["_masked", ""]
    elif kind == "mask":
        suffixes = ["_mask", ""]
    for suf in suffixes:
        for ext in exts:
            p = folder / f"{image_id}{suf}{ext}"
            if p.exists():
                return p
    for p in folder.iterdir():
        if p.suffix.lower() in exts and normalize_id(p.stem) == str(image_id):
            return p
    return None


def symlink_or_copy(src: Path, dst: Path, overwrite: bool = True):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return
    try:
        os.symlink(str(src.resolve()), str(dst))
    except Exception:
        shutil.copy2(str(src), str(dst))


def make_limited_dataset(dataset_root: Path, ids: List[str], limited_root: Path):
    mapping = {
        "test_GT": ("image", ""),
        "test_masked": ("masked", "_masked"),
        "test_mask": ("mask", "_mask"),
        "test_edge": ("image", "_edge"),
    }
    limited_root.mkdir(parents=True, exist_ok=True)

    for subdir, (kind, suffix) in mapping.items():
        src_dir = dataset_root / subdir
        if not src_dir.exists():
            continue
        dst_dir = limited_root / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)

        for image_id in ids:
            src = find_file_by_id(src_dir, image_id, kind if subdir != "test_edge" else "image")
            if src is None:
                # edge files may be named id_edge.jpg
                for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                    p = src_dir / f"{image_id}_edge{ext}"
                    if p.exists():
                        src = p
                        break
            if src is None:
                continue
            # Preserve original filename pattern for each folder.
            dst = dst_dir / src.name
            symlink_or_copy(src, dst)

    return limited_root


def get_prompt(item: Dict, mode: str) -> str:
    if not isinstance(item, dict):
        return str(item)
    prompts = item.get("prompts_for_ablation", {})
    if isinstance(prompts, dict) and mode in prompts:
        return str(prompts[mode])
    if mode == "full_structured":
        if "final_prompt" in item:
            return str(item["final_prompt"])
        if "prompt" in item:
            return str(item["prompt"])
    return "Dunhuang mural fragment, mineral color palette, flat contour lines, masked surface loss, boundary discontinuity."


def create_flat_prompt_json(prompt_db: Dict, mode: str, ids: List[str], out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for image_id in ids:
        prompt = get_prompt(prompt_db[str(image_id)], mode)
        flat[str(image_id)] = {"final_prompt": prompt, "prompt": prompt}
    out_file.write_text(json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_file


def patch_yml(base_text: str, name: str, prompt_path: Path, final_weight: str, dataset_root: Path, effective_dataset_root: Path) -> str:
    s = base_text

    if re.search(r"(?m)^name:\s*.*$", s):
        s = re.sub(r"(?m)^name:\s*.*$", f"name: {name}", s)
    else:
        s = f"name: {name}\n" + s

    if final_weight:
        s = re.sub(
            r"(?m)^(\s*)pretrain_model_G:\s*.*$",
            rf"\1pretrain_model_G: {final_weight}",
            s,
        )

    if re.search(r"(?m)^(\s*)prompt_path:\s*.*$", s):
        s = re.sub(
            r"(?m)^(\s*)prompt_path:\s*.*$",
            rf"\1prompt_path: {prompt_path}",
            s,
        )
    else:
        # Add under root if prompt_path does not exist.
        s += f"\n# Added by create_prompt_ablation_files.py\nprompt_path: {prompt_path}\n"

    # If a limited dataset is requested, replace the root string in every dataroot.
    if dataset_root and effective_dataset_root and str(dataset_root.resolve()) != str(effective_dataset_root.resolve()):
        s = s.replace(str(dataset_root.resolve()), str(effective_dataset_root.resolve()))
        s = s.replace(str(dataset_root), str(effective_dataset_root))

    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-yml", required=True)
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exp-prefix", default="test_combined_routeA_prompt_ablation")
    parser.add_argument("--final-weight", default="")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--limited-root", default="")
    args = parser.parse_args()

    base_yml = Path(args.base_yml)
    prompt_json = Path(args.prompt_json)
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(prompt_json, "r", encoding="utf-8") as f:
        prompt_db = json.load(f)

    ids = sort_ids(list(prompt_db.keys()))
    if args.limit and args.limit > 0:
        ids = ids[: args.limit]

    effective_dataset_root = dataset_root
    if args.limit and args.limit > 0:
        limited_root = Path(args.limited_root) if args.limited_root else dataset_root.parent / f"{dataset_root.name}_ablation_{args.limit}"
        effective_dataset_root = make_limited_dataset(dataset_root, ids, limited_root)
        print(f"[INFO] Limited dataset root: {effective_dataset_root}")

    modes = parse_modes(args.modes)
    base_text = base_yml.read_text(encoding="utf-8")

    yml_files = []
    prompt_dir = out_dir / "prompts"
    yml_dir = out_dir / "ymls"
    yml_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        flat_prompt = prompt_dir / f"prompts_{mode}.json"
        create_flat_prompt_json(prompt_db, mode, ids, flat_prompt)

        name = f"{args.exp_prefix}_{mode}"
        if args.limit and args.limit > 0:
            name += f"_N{args.limit}"

        yml_text = patch_yml(
            base_text=base_text,
            name=name,
            prompt_path=flat_prompt.resolve(),
            final_weight=args.final_weight,
            dataset_root=dataset_root,
            effective_dataset_root=effective_dataset_root,
        )

        yml_file = yml_dir / f"{name}.yml"
        yml_file.write_text(yml_text, encoding="utf-8")
        yml_files.append(yml_file)

    commands_file = out_dir / "run_commands.txt"
    with open(commands_file, "w", encoding="utf-8") as f:
        for yml in yml_files:
            f.write(f"python test_combined_final.py -opt {yml.resolve()}\n")

    print(f"[INFO] Wrote {len(yml_files)} yml files to {yml_dir}")
    print(f"[INFO] Wrote commands to {commands_file}")
    print("[INFO] First commands:")
    print(commands_file.read_text(encoding="utf-8").splitlines()[:3])


if __name__ == "__main__":
    main()
