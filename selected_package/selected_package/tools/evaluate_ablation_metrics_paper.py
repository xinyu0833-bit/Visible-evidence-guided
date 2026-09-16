# -*- coding: utf-8 -*-
"""
Paper-ready evaluator for Dunhuang inpainting ablations.

Metrics:
- Global PSNR / SSIM
- Hole PSNR / SSIM
- Boundary PSNR / SSIM
- Boundary Edge-F1
- LPIPS, optional
- NIQE, optional

Mask convention by default:
  white mask pixels = missing / hole region.

Boundary region:
  inner boundary band inside the hole: hole - erode(hole).
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as sk_ssim

try:
    import torch
except Exception:
    torch = None

try:
    import lpips
except Exception:
    lpips = None

try:
    import pyiqa
except Exception:
    pyiqa = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULT_ORDER = [
    "official_strdiffusion",
    "full_structured",
    "full",
    "content_degradation",
    "content_only",
    "degradation_only",
    "wrong_degradation",
    "random_prompt",
    "generic_prompt",
    "tag_prompt",
    "no_prompt",
    "null_prompt",
    "no_boundary",
    "no_routing_global",
    "hole_only_routing",
]


def normalize_id(stem: str) -> str:
    suffixes = [
        "_masked", "_mask", "_edge", "_pred", "_output", "_out", "_fake", "_sr",
        "-masked", "-mask", "-pred", "-output",
    ]
    s = stem
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if s.endswith(suf):
                s = s[: -len(suf)]
                changed = True
    return s


def read_rgb(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size and size > 0:
        img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def read_mask(path: Path, size: int, white_is_hole: bool = True, threshold: int = 127) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if size and size > 0:
        mask = mask.resize((size, size), Image.NEAREST)
    arr = np.asarray(mask)
    hole = arr > threshold if white_is_hole else arr <= threshold
    return hole.astype(bool)


def find_by_id(folder: Path, image_id: str, kind: str) -> Optional[Path]:
    if not folder.exists():
        return None
    candidates = []
    if kind == "mask":
        candidates += [f"{image_id}_mask", f"{image_id}"]
    else:
        candidates += [image_id, f"{image_id}_gt", f"{image_id}_GT"]
    for stem in candidates:
        for ext in IMAGE_EXTS:
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and normalize_id(p.stem) == str(image_id):
            return p
    return None


def psnr(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is not None:
        if mask.sum() == 0:
            return float("nan")
        diff = (a - b)[mask]
    else:
        diff = (a - b).reshape(-1, a.shape[-1])
    mse = float(np.mean(diff ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def ssim_scalar_and_map(a: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray]:
    try:
        val, smap = sk_ssim(a, b, data_range=1.0, channel_axis=2, full=True)
    except TypeError:
        val, smap = sk_ssim(a, b, data_range=1.0, multichannel=True, full=True)
    if smap.ndim == 3:
        smap = smap.mean(axis=2)
    return float(val), smap.astype(np.float32)


def masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(arr[mask].mean())


def inner_boundary_mask(hole: np.ndarray, kernel_size: int) -> np.ndarray:
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    hole_u = hole.astype(np.uint8)
    eroded = cv2.erode(hole_u, kernel, iterations=1).astype(bool)
    boundary = np.logical_and(hole, np.logical_not(eroded))
    if boundary.sum() == 0:
        boundary = hole
    return boundary


def edge_f1(pred: np.ndarray, gt: np.ndarray, region: np.ndarray, low: int, high: int, tolerance: int) -> float:
    pred_gray = cv2.cvtColor((np.clip(pred, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gt_gray = cv2.cvtColor((np.clip(gt, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    pe = cv2.Canny(pred_gray, low, high).astype(bool)
    ge = cv2.Canny(gt_gray, low, high).astype(bool)

    pe = np.logical_and(pe, region)
    ge = np.logical_and(ge, region)

    if pe.sum() == 0 and ge.sum() == 0:
        return float("nan")
    if pe.sum() == 0 or ge.sum() == 0:
        return 0.0

    t = max(0, int(tolerance))
    if t > 0:
        kernel = np.ones((2 * t + 1, 2 * t + 1), np.uint8)
        ge_d = cv2.dilate(ge.astype(np.uint8), kernel, iterations=1).astype(bool)
        pe_d = cv2.dilate(pe.astype(np.uint8), kernel, iterations=1).astype(bool)
    else:
        ge_d, pe_d = ge, pe

    precision = float(np.logical_and(pe, ge_d).sum()) / float(pe.sum() + 1e-8)
    recall = float(np.logical_and(ge, pe_d).sum()) / float(ge.sum() + 1e-8)
    if precision + recall <= 1e-12:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def load_lpips_model(no_lpips: bool):
    if no_lpips:
        return None
    if lpips is None or torch is None:
        print("[WARN] lpips or torch is unavailable. LPIPS will be NaN.")
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model


def compute_lpips(model, pred: np.ndarray, gt: np.ndarray) -> float:
    if model is None or torch is None:
        return float("nan")
    device = next(model.parameters()).device
    with torch.no_grad():
        p = torch.from_numpy(pred.transpose(2, 0, 1)).float().unsqueeze(0).to(device) * 2 - 1
        g = torch.from_numpy(gt.transpose(2, 0, 1)).float().unsqueeze(0).to(device) * 2 - 1
        return float(model(p, g).item())


def load_niqe_model(no_niqe: bool):
    if no_niqe:
        return None
    if pyiqa is None or torch is None:
        print("[WARN] pyiqa or torch is unavailable. NIQE will be NaN.")
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = pyiqa.create_metric("niqe", device=device)
    return model


def compute_niqe(model, pred: np.ndarray) -> float:
    if model is None or torch is None:
        return float("nan")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        p = torch.from_numpy(pred.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        return float(model(p).item())


def collect_method_dirs(pred_root: Path) -> Dict[str, Path]:
    # If pred_root itself contains image files, evaluate it as a single method.
    direct_images = [p for p in pred_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS] if pred_root.exists() else []
    if direct_images:
        return {pred_root.name: pred_root}

    methods = {}
    for d in sorted(pred_root.iterdir()):
        if d.is_dir() and not d.name.startswith("eval"):
            imgs = list(d.rglob("*"))
            if any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in imgs):
                methods[d.name] = d
    return methods


def collect_pred_files(method_dir: Path) -> Dict[str, Path]:
    files = {}
    for p in method_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            image_id = normalize_id(p.stem)
            # Prefer shallow path if duplicate.
            if image_id not in files or len(p.parts) < len(files[image_id].parts):
                files[image_id] = p
    return files


def mask_group(mask_ratio: float) -> str:
    pct = mask_ratio * 100.0
    if pct < 20:
        return "0-20%"
    if pct < 40:
        return "20-40%"
    if pct < 60:
        return "40-60%"
    return "60-100%"


def order_methods(df: pd.DataFrame, method_col: str = "Method") -> pd.DataFrame:
    order = {m: i for i, m in enumerate(DEFAULT_ORDER)}
    def key(m):
        s = str(m)
        for name, idx in order.items():
            if s == name or name in s:
                return idx
        return 999
    return df.assign(_order=df[method_col].map(key)).sort_values(["_order", method_col]).drop(columns=["_order"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--boundary-kernel", type=int, default=5)
    parser.add_argument("--mask-white-is-hole", action="store_true", default=True)
    parser.add_argument("--mask-black-is-hole", dest="mask_white_is_hole", action="store_false")
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    parser.add_argument("--edge-tolerance", type=int, default=2)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--no-niqe", action="store_true")
    args = parser.parse_args()

    pred_root = Path(args.pred_root)
    gt_dir = Path(args.gt_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = collect_method_dirs(pred_root)
    if not methods:
        raise SystemExit(f"No method image directories found under {pred_root}")

    lpips_model = load_lpips_model(args.no_lpips)
    niqe_model = load_niqe_model(args.no_niqe)

    rows = []
    for method, mdir in methods.items():
        pred_files = collect_pred_files(mdir)
        print(f"[INFO] {method}: {len(pred_files)} prediction images")
        for image_id, pred_path in tqdm(sorted(pred_files.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0])), desc=method):
            gt_path = find_by_id(gt_dir, image_id, "gt")
            mask_path = find_by_id(mask_dir, image_id, "mask")
            if gt_path is None or mask_path is None:
                continue

            pred = read_rgb(pred_path, args.eval_size)
            gt = read_rgb(gt_path, args.eval_size)
            hole = read_mask(mask_path, args.eval_size, args.mask_white_is_hole)
            boundary = inner_boundary_mask(hole, args.boundary_kernel)

            g_ssim, ssim_map = ssim_scalar_and_map(pred, gt)

            row = {
                "Method": method,
                "ImageID": image_id,
                "PredPath": str(pred_path),
                "MaskRatio": float(hole.mean()),
                "MaskGroup": mask_group(float(hole.mean())),
                "G-PSNR": psnr(pred, gt),
                "G-SSIM": g_ssim,
                "H-PSNR": psnr(pred, gt, hole),
                "H-SSIM": masked_mean(ssim_map, hole),
                "B-PSNR": psnr(pred, gt, boundary),
                "B-SSIM": masked_mean(ssim_map, boundary),
                "B-EdgeF1": edge_f1(pred, gt, boundary, args.canny_low, args.canny_high, args.edge_tolerance),
                "LPIPS": compute_lpips(lpips_model, pred, gt),
                "NIQE": compute_niqe(niqe_model, pred),
            }
            rows.append(row)

    if not rows:
        raise SystemExit("No matched prediction/GT/mask samples found.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "all_per_image_metrics.csv", index=False)

    metric_cols = ["G-PSNR", "G-SSIM", "H-PSNR", "H-SSIM", "B-PSNR", "B-SSIM", "B-EdgeF1", "LPIPS", "NIQE"]

    summary = df.groupby("Method")[metric_cols].mean(numeric_only=True).reset_index()
    summary = order_methods(summary)
    summary.to_csv(out_dir / "paper_table_all.csv", index=False)

    by_mask = df.groupby(["MaskGroup", "Method"])[metric_cols].mean(numeric_only=True).reset_index()
    by_mask = by_mask.sort_values(["MaskGroup", "Method"])
    by_mask.to_csv(out_dir / "paper_table_by_mask.csv", index=False)

    std = df.groupby("Method")[metric_cols].std(numeric_only=True).reset_index()
    std = order_methods(std)
    std.to_csv(out_dir / "paper_table_std.csv", index=False)

    try:
        (out_dir / "paper_table_all.md").write_text(summary.to_markdown(index=False), encoding="utf-8")
        (out_dir / "paper_table_by_mask.md").write_text(by_mask.to_markdown(index=False), encoding="utf-8")
    except Exception:
        pass

    print(f"[INFO] Saved evaluation to {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
