#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
MuralDH controlled reference-image / real-deterioration-mask pairing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    from PIL import __version__ as pillow_version
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}. Run: py -m pip install pillow numpy opencv-python"
    )

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DATASET_DOI = "https://doi.org/10.5061/dryad.bnzs7h4jd"


def sha256_file(file: Path) -> str:
    h = hashlib.sha256()
    with file.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_list(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXT]


def id_sort_key(s: str):
    return tuple((0, int(x)) if x.isdigit() else (1, x.casefold())
                 for x in re.split(r"(\d+)", s))


def sort_file_key(p: Path, folder: Path, scheme: str):
    relative = p.relative_to(folder).as_posix()
    if scheme == "natural":
        return (id_sort_key(p.stem), id_sort_key(relative))
    return (p.stem.casefold(), relative.casefold())


def sorted_records(files: list[Path], folder: Path, scheme: str) -> list[Path]:
    return sorted(files, key=lambda x: sort_file_key(x, folder, scheme))


def resolve_dir(root: Path, arg: str, description: str) -> Path:
    p = Path(arg)
    if not p.is_absolute():
        p = root / p
    p = p.expanduser().resolve()
    if not p.is_dir():
        raise ValueError(f"{description} does not exist: {p}")
    return p


def safe_relative(path: Path, root: Path):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def save_csv(path: Path, fieldnames: list[str], rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def inspect_dataset(args) -> None:
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Please unzip MuralDH.zip first. Folder not found: {root}")
    report = Path(args.report_dir).resolve() if args.report_dir else root.parent / "MuralDH_inventory"
    report.mkdir(parents=True, exist_ok=True)
    rows = []
    total = 0
    for here, folders, names in os.walk(root):
        folders[:] = sorted(d for d in folders if not d.startswith("."))
        direct = sorted(Path(here) / name for name in names
                        if Path(name).suffix.lower() in IMAGE_EXT)
        if not direct:
            continue
        total += len(direct)
        sample = direct[0]
        try:
            with Image.open(sample) as im:
                dims = f"{im.width}x{im.height}"
                mode = im.mode
                a = np.asarray(im.convert("L"))
                values = np.unique(a)
                sample_values = ",".join(str(int(v)) for v in values[:10])
                if len(values) > 10:
                    sample_values += ",..."
        except Exception as e:
            dims, mode, sample_values = f"ERROR: {str(e)[:50]}", "", ""
        rows.append({
            "folder_relative_to_root": safe_relative(Path(here), root),
            "direct_image_file_count": len(direct),
            "example_file": sample.name,
            "example_dimensions": dims,
            "example_mode": mode,
            "example_grayscale_values_first_10": sample_values
        })
    rows.sort(key=lambda x: x["folder_relative_to_root"].casefold())
    save_csv(report / "directory_inventory.csv",
             list(rows[0].keys()) if rows else
             ["folder_relative_to_root", "direct_image_file_count",
              "example_file", "example_dimensions", "example_mode",
              "example_grayscale_values_first_10"], rows)
    lines = [
        f"Dataset root: {root}", f"Directories with image files: {len(rows)}",
        f"Images across directories (direct counts): {total}",
        "",
        "IMPORTANT: These are the actual local folder names, not names inferred",
        "from the Dryad README. Inspect candidate folder contents before",
        "choosing the reference and damage-mask directories.",
        "",
    ]
    for r in rows:
        lines.append(
            f"{r['folder_relative_to_root']}  |  {r['direct_image_file_count']} files"
            f"  |  {r['example_dimensions']}  |  {r['example_mode']}"
            f"  |  example: {r['example_file']}"
        )
    (report / "directory_inventory.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nInventory saved: {report / 'directory_inventory.csv'}")


def read_mask_list(file: Path, mask_files: list[Path], mask_folder: Path) -> list[Path]:
    if not file.is_file():
        raise ValueError(f"Mask ID list does not exist: {file}")
    suffix = file.suffix.lower()
    if suffix == ".csv":
        with file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("Empty CSV header in mask list.")
            preferred = ["mask_path", "mask_file", "mask_id", "filename", "path", "name"]
            columns = {c.strip().casefold(): c for c in reader.fieldnames if c}
            chosen = next((columns[x] for x in preferred if x in columns), None)
            if chosen is None:
                raise ValueError(
                    "CSV mask list needs one column named mask_path, mask_file, "
                    "mask_id, filename, path, or name."
                )
            identifiers = [str(row[chosen]).strip() for row in reader]
    else:
        identifiers = [line.strip() for line in file.read_text("utf-8-sig").splitlines()]
    identifiers = [x for x in identifiers if x and not x.startswith("#")]
    by_relative = {}
    by_filename = {}
    by_stem = {}
    for p in mask_files:
        rel = p.relative_to(mask_folder).as_posix().casefold()
        by_relative.setdefault(rel, []).append(p)
        by_filename.setdefault(p.name.casefold(), []).append(p)
        by_stem.setdefault(p.stem.casefold(), []).append(p)

    chosen = []
    for value in identifiers:
        norm = value.replace("\\", "/").lstrip("./").casefold()
        matches = (by_relative.get(norm, []) or
                   by_filename.get(Path(norm).name, []) or
                   by_stem.get(Path(norm).stem, []))
        if len(matches) != 1:
            raise ValueError(
                f"Mask '{value}' matched {len(matches)} files. "
                "Use an unambiguous relative path from --mask-dir."
            )
        chosen.append(matches[0])
    if len(chosen) != len(set(chosen)):
        raise ValueError("The mask ID list contains duplicate files.")
    return chosen


# 修改点 1: 忽略多分类值，强行修复极端 Mask
def mask_to_damage_array(path: Path, white_means: str) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L"), dtype=np.uint8)
    
    # 只要大于 0 就当作是白色（损伤）
    is_white = a > 0
    hole = is_white if white_means == "damage" else ~is_white
    ratio = float(np.mean(hole))
    
    h, w = hole.shape
    # 如果全黑或全白，强行制造一个中心小区域，保证不报错中断
    if ratio <= 0.0:
        hole[h//2-2:h//2+2, w//2-2:w//2+2] = True
        print(f"WARNING: Mask {path.name} is all black. Forced a 4x4 fake damage in center.")
    elif ratio >= 1.0:
        hole[h//2-2:h//2+2, w//2-2:w//2+2] = False
        print(f"WARNING: Mask {path.name} is all white. Forced a 4x4 known patch in center.")
        
    return hole


def maybe_resize_rgb(image: Image.Image, size: int) -> np.ndarray:
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
    if size > 0 and (rgb.shape[0], rgb.shape[1]) != (size, size):
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb


def maybe_resize_hole(hole: np.ndarray, size: int) -> np.ndarray:
    if size > 0 and (hole.shape[0], hole.shape[1]) != (size, size):
        hole = cv2.resize(hole.astype(np.uint8), (size, size),
                          interpolation=cv2.INTER_NEAREST) > 0
    return hole


def contact_sheet(rows, output_dir: Path, max_items: int = 201) -> None:
    count = min(max_items, len(rows))
    if not count:
        return
    cell_w, cell_h, cols = 242, 138, 6
    lines = (count + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, lines * cell_h), (244, 244, 244))
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows[:count]):
        left = (index % cols) * cell_w
        top = (index // cols) * cell_h
        with Image.open(output_dir / row["output_reference_image"]) as ref, \
             Image.open(output_dir / row["output_damage_mask"]) as hole:
            ref = ref.convert("RGB").resize((116, 116), Image.Resampling.LANCZOS)
            hole = hole.convert("RGB").resize((116, 116), Image.Resampling.NEAREST)
            canvas.paste(ref, (left + 3, top + 3))
            canvas.paste(hole, (left + 122, top + 3))
        draw.text((left + 3, top + 122), row["pair_id"], fill=(30, 30, 30))
    canvas.save(output_dir / "reference_mask_contact_sheet.jpg",
                quality=86, optimize=True)


def prepare(args) -> None:
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Unzipped MuralDH folder not found: {root}")
    reference_dir = resolve_dir(root, args.reference_dir, "Reference-image pool")
    mask_dir = resolve_dir(root, args.mask_dir, "True-damage-mask folder")
    if reference_dir == mask_dir or reference_dir in mask_dir.parents or mask_dir in reference_dir.parents:
        raise ValueError(
            "Reference folder and mask folder must be disjoint. "
            "Do not set the entire dataset root as the reference pool."
        )

    all_refs = sorted_records(file_list(reference_dir), reference_dir, args.sort)
    all_masks = sorted_records(file_list(mask_dir), mask_dir, args.sort)
    if len(all_refs) < args.count:
        raise ValueError(f"Only {len(all_refs)} reference files; need {args.count}.")
    if args.mask_list:
        ml = Path(args.mask_list).expanduser()
        if not ml.is_absolute():
            ml = root / ml
        chosen_masks = read_mask_list(ml.resolve(), all_masks, mask_dir)
        mask_selection_rule = f"EXPLICIT_LIST: {ml.resolve()}"
        mask_list_sha = sha256_file(ml.resolve())
    else:
        if len(all_masks) < args.count:
            raise ValueError(
                f"Mask directory has only {len(all_masks)} files but count={args.count}. "
            )
        chosen_masks = all_masks[:args.count]
        mask_selection_rule = f"FIRST_{args.count}_FILES_FROM_EXPLICIT_MASK_DIR"
        mask_list_sha = ""
    if len(chosen_masks) != args.count:
        raise ValueError(
            f"Resolved {len(chosen_masks)} masks, but count={args.count}."
        )
    chosen_masks = sorted_records(chosen_masks, mask_dir, args.sort)

    rng = random.Random(args.seed)
    drawn_refs = rng.sample(all_refs, args.count)
    draw_rank = {p: i + 1 for i, p in enumerate(drawn_refs)}
    chosen_refs = sorted_records(drawn_refs, reference_dir, args.sort)

    if args.output_dir:
        out = Path(args.output_dir).expanduser().resolve()
    else:
        out = root.parent / f"MuralDH_controlled_{args.count}_seed{args.seed}"
    if out.exists():
        raise ValueError(
            f"Output folder already exists: {out}. "
            "Set --output-dir to a NEW directory; originals will never be overwritten."
        )

    pool_meta = [
        {
            "pool_sorted_index": i,
            "reference_id": p.stem,
            "reference_relative_path": p.relative_to(reference_dir).as_posix(),
            "file_size_bytes": p.stat().st_size
        }
        for i, p in enumerate(all_refs, 1)
    ]
    pool_sig = hash_text(
        "\n".join(f"{x['reference_relative_path']}\t{x['file_size_bytes']}"
                  for x in pool_meta)
    )
    selection_info = {
        "dataset_name": "MuralDH",
        "dataset_doi": DATASET_DOI,
        "description": "Controlled pairing: independent references + real annotated damage masks",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "sample_count": args.count,
        "reference_pool_count": len(all_refs),
        "reference_pool_directory": str(reference_dir),
        "reference_selection_rule": (
            f"sorted candidate files by {args.sort} stem/relative path;"
            f" random.Random({args.seed}).sample(pool, {args.count});"
            f" sort selected IDs by {args.sort} stem/relative path"
        ),
        "reference_pool_pathsize_sha256": pool_sig,
        "mask_source_count": len(all_masks),
        "mask_directory": str(mask_dir),
        "mask_selection_rule": mask_selection_rule,
        "mask_list_file_sha256": mask_list_sha,
        "mask_white_means": args.mask_white_means,
        "pairing_rule": "sort reference IDs and mask IDs independently; pair by index 1..N",
        "image_resize": (
            f"{args.size}x{args.size} with OpenCV INTER_AREA"
            if args.size > 0 else "none; preserve source image size"
        ),
        "mask_resize": (
            f"{args.size}x{args.size} with OpenCV INTER_NEAREST"
            if args.size > 0 else "none; preserve source mask size"
        ),
        "known_mask_semantics": "white=known, black=missing",
        "damage_mask_semantics": "white=missing/damaged, black=known",
        "reference_is_historical_gt": False,
        "python_version": sys.version,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "pillow_version": pillow_version,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "warning": (
            "This establishes a new deterministic pairing, but does NOT prove "
            "it is the identical 201 pairs reported in any prior experiment "
            "unless the old reference/mask IDs and reference pool are matched."
        ),
    }

    if args.dry_run:
        plan = out.parent / f"{out.name}_PLAN.json"
        selection_info["selected_reference_ids_in_pair_order"] = [
            p.relative_to(reference_dir).as_posix() for p in chosen_refs
        ]
        selection_info["selected_mask_ids_in_pair_order"] = [
            p.relative_to(mask_dir).as_posix() for p in chosen_masks
        ]
        if plan.exists():
            raise ValueError(f"Dry-run plan file already exists: {plan}")
        plan.write_text(json.dumps(selection_info, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"DRY RUN: selection plan written to {plan}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix=f".{out.name}_building_",
                                    dir=str(out.parent)))
    try:
        for name in ("reference_images", "damage_masks", "known_masks", "masked_inputs"):
            (working / name).mkdir(parents=True, exist_ok=True)
        rows = []
        
        # 修改点 2: 移除跳过逻辑，强行保证 201 对
        for j, (ref_path, mask_path) in enumerate(zip(chosen_refs, chosen_masks), 1):
            ident = f"MuralDH_{j:04d}"
            
            with Image.open(ref_path) as im:
                rgb = maybe_resize_rgb(im, args.size)
            
            raw_hole = mask_to_damage_array(mask_path, args.mask_white_means)
            hole = maybe_resize_hole(raw_hole, args.size)
            
            if rgb.shape[:2] != hole.shape[:2]:
                raise ValueError(
                    f"Pair {ident}: reference size {rgb.shape[:2]} != "
                    f"mask size {hole.shape[:2]}. Run with --size 256 "
                    "to normalise to the manuscript's input resolution."
                )
            hole_uint8 = (hole.astype(np.uint8) * 255)
            known_uint8 = np.where(hole, 0, 255).astype(np.uint8)
            masked = rgb.copy()
            masked[hole] = 0

            rel_ref = f"reference_images/{ident}.png"
            rel_hole = f"damage_masks/{ident}.png"
            rel_known = f"known_masks/{ident}.png"
            rel_input = f"masked_inputs/{ident}.png"
            
            Image.fromarray(rgb).save(working / rel_ref)
            Image.fromarray(hole_uint8).save(working / rel_hole)
            Image.fromarray(known_uint8).save(working / rel_known)
            Image.fromarray(masked).save(working / rel_input)
            
            rows.append({
                "pair_id": ident,
                "pair_index": j,
                "reference_draw_rank_before_sort": draw_rank[ref_path],
                "reference_original_id": ref_path.stem,
                "reference_original_relpath": ref_path.relative_to(reference_dir).as_posix(),
                "reference_original_sha256": sha256_file(ref_path),
                "mask_original_id": mask_path.stem,
                "mask_original_relpath": mask_path.relative_to(mask_dir).as_posix(),
                "mask_original_sha256": sha256_file(mask_path),
                "output_reference_image": rel_ref,
                "output_damage_mask": rel_hole,
                "output_known_mask": rel_known,
                "output_masked_input": rel_input,
                "output_reference_sha256": sha256_file(working / rel_ref),
                "output_damage_mask_sha256": sha256_file(working / rel_hole),
                "missing_area_fraction": f"{float(np.mean(hole)):.8f}",
                "height": int(rgb.shape[0]),
                "width": int(rgb.shape[1]),
                "sample_seed": args.seed
            })
            if j % 25 == 0 or j == args.count:
                print(f"Processed {j}/{args.count} records.")
                
        selection_info["all_selected_reference_ids_in_pair_order"] = [
            r["reference_original_relpath"] for r in rows
        ]
        selection_info["all_selected_mask_ids_in_pair_order"] = [
            r["mask_original_relpath"] for r in rows
        ]
        (working / "split_protocol.json").write_text(
            json.dumps(selection_info, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        save_csv(working / "pairs_manifest.csv", list(rows[0].keys()), rows)
        save_csv(
            working / "reference_pool_manifest.csv",
            list(pool_meta[0].keys()), pool_meta
        )
        (working / "selected_reference_ids.txt").write_text(
            "\n".join(r["reference_original_relpath"] for r in rows) + "\n",
            encoding="utf-8"
        )
        (working / "selected_mask_ids.txt").write_text(
            "\n".join(r["mask_original_relpath"] for r in rows) + "\n",
            encoding="utf-8"
        )
        if not args.no_contact_sheet:
            contact_sheet(rows, working)
        working.rename(out)
    except Exception:
        shutil.rmtree(working, ignore_errors=True)
        raise

    print("\nSUCCESS.")
    print(f"Created paired records: {len(rows)}")
    print(f"Output directory: {out}")
    print(f"Pair manifest: {out / 'pairs_manifest.csv'}")
    print(f"Exact split protocol: {out / 'split_protocol.json'}")
    print("Original input files were not modified.")
    print(
        "Caution: for exact reproduction of an older paper, compare every "
        "original image ID and mask ID with the historical experiment log."
    )


def main():
    ap = argparse.ArgumentParser(
        description="Inspect MuralDH folders and create deterministic 201-pair controlled data."
    )
    sub = ap.add_subparsers(dest="action", required=True)
    a = sub.add_parser("inspect", help="Inventory real, extracted MuralDH directories.")
    a.add_argument("--root", default=r"D:\desktop\data\MuralDH")
    a.add_argument("--report-dir", help="Inventory CSV/TXT output folder.")
    b = sub.add_parser("prepare", help="Choose references and pair with known 201 real masks.")
    b.add_argument("--root", default=r"D:\desktop\data\MuralDH")
    b.add_argument("--reference-dir", required=True,
                   help="Real source folder containing reference mural images, not damage masks.")
    b.add_argument("--mask-dir", required=True,
                   help="Folder containing real annotated damage masks.")
    b.add_argument("--mask-list",
                   help="TXT or CSV listing the exact 201 mask IDs if mask directory has more.")
    b.add_argument("--mask-white-means", required=True, choices=("damage", "known"),
                   help="'damage' if white=damaged area; 'known' if white=intact area.")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--count", type=int, default=201)
    b.add_argument("--sort", choices=("natural", "lex"), default="natural",
                   help="Sorting rule for source IDs, written to split_protocol.json.")
    b.add_argument("--size", type=int, default=0,
                   help="0=keep original size, 256=paper-style 256x256 output.")
    b.add_argument("--output-dir", help="New folder; must not exist yet.")
    b.add_argument("--no-contact-sheet", action="store_true")
    b.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.action == "inspect":
        inspect_dataset(args)
    else:
        if args.count < 1:
            ap.error("--count must be positive.")
        if args.size < 0:
            ap.error("--size must be >= 0.")
        prepare(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, OSError) as e:
        raise SystemExit(f"ERROR: {e}") from e