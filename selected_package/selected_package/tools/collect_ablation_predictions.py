# -*- coding: utf-8 -*-
"""
Collect predictions from multiple separate test result folders into a single
method-directory layout for evaluation.

Target layout:
  output_root/
    full_structured/
      501.png
      502.png
    wrong_degradation/
      501.png
      502.png

This is useful when the existing test_combined_final.py writes each yml's
results into a separate folder under results/.
"""

import argparse
import os
import shutil
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_modes(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def normalize_mode(mode):
    aliases = {
        "full": "full_structured",
        "full_prompt": "full_structured",
    }
    return aliases.get(mode, mode)


def find_candidate_dirs(results_root: Path, mode: str, image_dir_name: str):
    candidates = []
    for d in results_root.rglob(image_dir_name):
        if not d.is_dir():
            continue
        pstr = str(d)
        if mode in pstr or normalize_mode(mode) in pstr:
            if any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in d.rglob("*")):
                candidates.append(d)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def symlink_or_copy(src: Path, dst: Path, copy: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(str(src), str(dst))
    else:
        try:
            os.symlink(str(src.resolve()), str(dst))
        except Exception:
            shutil.copy2(str(src), str(dst))


def collect_images(src_dir: Path, dst_dir: Path, copy: bool = False):
    n = 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            # Flatten. If duplicate names exist, keep the shortest/deepest-safe unique name.
            dst = dst_dir / p.name
            if dst.exists() or dst.is_symlink():
                dst = dst_dir / f"{p.parent.name}_{p.name}"
            symlink_or_copy(p, dst, copy=copy)
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--modes", required=True)
    parser.add_argument("--image-dir-name", default="images_hard")
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    modes = parse_modes(args.modes)
    for mode in modes:
        candidates = find_candidate_dirs(results_root, mode, args.image_dir_name)
        if not candidates and mode == "full_structured":
            candidates = find_candidate_dirs(results_root, "full", args.image_dir_name)
        if not candidates:
            print(f"[WARN] No {args.image_dir_name} result dir found for mode={mode}")
            continue

        src = candidates[0]
        dst = output_root / mode
        if dst.exists():
            shutil.rmtree(dst)
        n = collect_images(src, dst, copy=args.copy)
        print(f"[INFO] {mode}: collected {n} images from {src} -> {dst}")

    print(f"[INFO] Collected root: {output_root}")


if __name__ == "__main__":
    main()
