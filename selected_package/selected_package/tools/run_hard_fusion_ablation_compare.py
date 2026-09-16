# -*- coding: utf-8 -*-
import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except Exception as e:
    raise RuntimeError("This script requires opencv-python. Please install or activate the correct env.") from e

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

try:
    import torch
    import lpips
except Exception:
    torch = None
    lpips = None


PROMPT_MODES = [
    "full_structured",
    "wrong_degradation",
    "random_prompt",
    "generic_prompt",
    "tag_prompt",
    "content_only",
    "degradation_only",
    "content_degradation",
    "no_prompt",
    "null_prompt",
    "empty_prompt",
    "full",
]


HIGHER_BETTER = {
    "G_PSNR": True,
    "G_SSIM": True,
    "H_PSNR": True,
    "H_SSIM": True,
    "B_PSNR": True,
    "B_SSIM": True,
    "B_EdgeF1": True,
    "LPIPS": False,
}


def sanitize_name(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x.strip("_")


def image_id_from_path(p):
    stem = Path(p).stem
    m = re.search(r"(\d+)", stem)
    return m.group(1) if m else None


def infer_method_from_yml(yml_path):
    stem = Path(yml_path).stem
    for mode in sorted(PROMPT_MODES, key=len, reverse=True):
        if mode in stem:
            return mode
    return sanitize_name(stem)


def list_images(folder):
    folder = Path(folder)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def build_id_map(folder):
    out = {}
    for p in list_images(folder):
        image_id = image_id_from_path(p)
        if image_id is not None and image_id not in out:
            out[image_id] = p
    return out


def read_rgb(path):
    return Image.open(path).convert("RGB")


def read_mask(path, size, mask_white_is_known=False):
    m = Image.open(path).convert("L")
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    arr = np.array(m).astype(np.float32) / 255.0
    arr = (arr > 0.5).astype(np.float32)

    # default: white mask means missing hole.
    # if white means known region, invert it to get hole mask.
    if mask_white_is_known:
        arr = 1.0 - arr

    return arr


def resize_rgb(img, size):
    if img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return img


def hard_fuse(pred_path, gt_path, mask_path, out_path, mask_white_is_known=False):
    gt_img = read_rgb(gt_path)
    pred_img = resize_rgb(read_rgb(pred_path), gt_img.size)

    gt = np.array(gt_img).astype(np.float32)
    pred = np.array(pred_img).astype(np.float32)
    hole = read_mask(mask_path, gt_img.size, mask_white_is_known=mask_white_is_known)

    hole3 = hole[:, :, None]
    fused = gt * (1.0 - hole3) + pred * hole3
    fused = np.clip(fused, 0, 255).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(fused).save(out_path)

    return out_path


def patch_yml_name(src_yml, out_yml, exp_name):
    text = Path(src_yml).read_text(encoding="utf-8")

    if re.search(r"(?m)^name:\s*.*$", text):
        text = re.sub(r"(?m)^name:\s*.*$", f"name: {exp_name}", text)
    else:
        text = f"name: {exp_name}\n" + text

    out_yml.parent.mkdir(parents=True, exist_ok=True)
    out_yml.write_text(text, encoding="utf-8")
    return out_yml


def run_test_for_yml(test_script, yml_path, work_dir, dry_run=False):
    cmd = [
        sys.executable,
        str(test_script),
        "-opt",
        str(yml_path),
    ]

    print("[RUN]", " ".join(cmd))

    if dry_run:
        return

    subprocess.run(cmd, cwd=str(work_dir), check=True)


def score_candidate_dir(d, gt_ids):
    files = list_images(d)
    ids = set()
    for p in files:
        s = str(p).lower()
        if any(bad in s for bad in ["mask", "gt", "edge", "input"]):
            continue
        image_id = image_id_from_path(p)
        if image_id in gt_ids:
            ids.add(image_id)

    score = len(ids)
    name = str(d).lower()

    bonus = 0
    if "images_hard" in name or "hard" in name:
        bonus += 1000
    elif "images" in name or "visual" in name:
        bonus += 100

    return score + bonus, score


def find_prediction_dir(results_root, exp_name, gt_ids):
    results_root = Path(results_root)
    candidates = []

    preferred_root = results_root / exp_name
    search_roots = []

    if preferred_root.exists():
        search_roots.append(preferred_root)

    for p in results_root.rglob("*"):
        if p.is_dir() and exp_name in str(p):
            search_roots.append(p)

    if not search_roots:
        search_roots = [results_root]

    seen = set()
    for root in search_roots:
        for f in list_images(root):
            parent = f.parent
            if parent in seen:
                continue
            seen.add(parent)
            total_score, match_count = score_candidate_dir(parent, gt_ids)
            if match_count > 0:
                candidates.append((total_score, match_count, parent))

    if not candidates:
        raise RuntimeError(f"Cannot find prediction images for exp_name={exp_name} under {results_root}")

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    print("[PRED_DIR]", exp_name, "=>", candidates[0][2], "matched:", candidates[0][1])
    return candidates[0][2]


def psnr_from_mse(mse):
    if mse is None or math.isnan(mse):
        return float("nan")
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def region_mse(a, b, region):
    region = region.astype(bool)
    if region.sum() == 0:
        return float("nan")
    diff = a.astype(np.float32) - b.astype(np.float32)
    diff = diff[region]
    return float(np.mean(diff ** 2))


def global_ssim(a, b):
    if skimage_ssim is None:
        return float("nan")
    return float(skimage_ssim(a, b, data_range=255, channel_axis=2))


def region_ssim(a, b, region):
    if skimage_ssim is None:
        return float("nan")

    try:
        _, ssim_map = skimage_ssim(
            a,
            b,
            data_range=255,
            channel_axis=2,
            full=True,
        )
    except Exception:
        return float("nan")

    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=2)

    region = region.astype(bool)
    if region.sum() == 0:
        return float("nan")

    return float(np.mean(ssim_map[region]))


def boundary_region(hole, kernel_size):
    k = np.ones((kernel_size, kernel_size), np.uint8)
    h = hole.astype(np.uint8)
    dil = cv2.dilate(h, k, iterations=1)
    ero = cv2.erode(h, k, iterations=1)
    b = (dil - ero) > 0
    return b


def edge_f1(pred, gt, region, low=100, high=200):
    region = region.astype(bool)
    if region.sum() == 0:
        return float("nan")

    pred_gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)

    ep = cv2.Canny(pred_gray, low, high) > 0
    eg = cv2.Canny(gt_gray, low, high) > 0

    ep = ep & region
    eg = eg & region

    tp = np.logical_and(ep, eg).sum()
    fp = np.logical_and(ep, ~eg).sum()
    fn = np.logical_and(~ep, eg).sum()

    if tp + fp + fn == 0:
        return 1.0

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def make_lpips_model(use_lpips):
    if not use_lpips:
        return None

    if torch is None or lpips is None:
        print("[WARN] lpips package is not available. LPIPS will be NaN.")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model


def lpips_score(model, pred, gt):
    if model is None:
        return float("nan")

    device = next(model.parameters()).device

    def to_tensor(x):
        x = torch.from_numpy(x).float() / 127.5 - 1.0
        x = x.permute(2, 0, 1).unsqueeze(0).to(device)
        return x

    with torch.no_grad():
        val = model(to_tensor(pred), to_tensor(gt)).item()
    return float(val)


def eval_one_image(pred_path, gt_path, mask_path, eval_size, boundary_kernel, mask_white_is_known, lpips_model):
    gt_img = read_rgb(gt_path)
    pred_img = resize_rgb(read_rgb(pred_path), gt_img.size)

    gt_img_eval = gt_img.resize((eval_size, eval_size), Image.BICUBIC)
    pred_img_eval = pred_img.resize((eval_size, eval_size), Image.BICUBIC)

    gt = np.array(gt_img_eval).astype(np.uint8)
    pred = np.array(pred_img_eval).astype(np.uint8)

    mask_img = Image.open(mask_path).convert("L").resize((eval_size, eval_size), Image.NEAREST)
    hole = (np.array(mask_img).astype(np.float32) / 255.0 > 0.5).astype(np.float32)

    if mask_white_is_known:
        hole = 1.0 - hole

    bound = boundary_region(hole, boundary_kernel)
    all_region = np.ones_like(hole).astype(bool)
    hole_region = hole.astype(bool)

    out = {}

    out["G_PSNR"] = psnr_from_mse(region_mse(pred, gt, all_region))
    out["H_PSNR"] = psnr_from_mse(region_mse(pred, gt, hole_region))
    out["B_PSNR"] = psnr_from_mse(region_mse(pred, gt, bound))

    out["G_SSIM"] = global_ssim(pred, gt)
    out["H_SSIM"] = region_ssim(pred, gt, hole_region)
    out["B_SSIM"] = region_ssim(pred, gt, bound)

    out["B_EdgeF1"] = edge_f1(pred, gt, bound)
    out["LPIPS"] = lpips_score(lpips_model, pred, gt)

    return out


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("[CSV]", path)


def mean_ignore_nan(vals):
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def summarize(per_image_rows, metric_names):
    methods = sorted(set(r["method"] for r in per_image_rows))
    rows = []

    for method in methods:
        subset = [r for r in per_image_rows if r["method"] == method]
        row = {"method": method, "num_images": len(subset)}

        for m in metric_names:
            row[m] = mean_ignore_nan([r[m] for r in subset])

        rows.append(row)

    return rows


def compare_to_reference(summary_rows, reference, metric_names):
    by_method = {r["method"]: r for r in summary_rows}

    if reference not in by_method:
        print(f"[WARN] reference={reference} not found. Use first method as reference.")
        reference = summary_rows[0]["method"]

    ref = by_method[reference]
    rows = []

    for r in summary_rows:
        method = r["method"]
        row = {
            "method": method,
            "reference": reference,
        }

        for m in metric_names:
            v = float(r[m])
            rv = float(ref[m])

            row[m] = v
            row[f"ref_{m}"] = rv

            if math.isnan(v) or math.isnan(rv):
                delta = float("nan")
                improved = ""
            else:
                delta = v - rv
                if HIGHER_BETTER.get(m, True):
                    improved = delta > 0
                else:
                    improved = delta < 0

            row[f"delta_{m}"] = delta
            row[f"better_{m}"] = improved

        rows.append(row)

    return rows


def pixel_diff_vs_reference(hard_root, methods, ids, reference):
    rows = []

    ref_dir = Path(hard_root) / reference
    if not ref_dir.exists():
        print(f"[WARN] reference hard dir not found: {ref_dir}")
        return rows

    for method in methods:
        if method == reference:
            continue

        method_dir = Path(hard_root) / method

        for image_id in ids:
            p1 = ref_dir / f"{image_id}.png"
            p2 = method_dir / f"{image_id}.png"

            if not p1.exists() or not p2.exists():
                continue

            a = np.array(Image.open(p1).convert("RGB")).astype(np.int16)
            b = np.array(Image.open(p2).convert("RGB")).astype(np.int16)

            if a.shape != b.shape:
                continue

            diff = np.abs(a - b)
            rows.append({
                "method": method,
                "reference": reference,
                "image_id": image_id,
                "max_abs_diff": int(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "pixel_identical": bool(np.array_equal(a, b)),
            })

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-script", default="test_combined_final.py")
    parser.add_argument("--yml-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--exp-prefix", default="hard_prompt_ablation")
    parser.add_argument("--reference", default="full_structured")
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--boundary-kernel", type=int, default=5)

    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--mask-white-is-known", action="store_true")
    parser.add_argument("--no-lpips", action="store_true")

    args = parser.parse_args()

    work_dir = Path.cwd()
    yml_dir = Path(args.yml_dir)
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)

    gt_map = build_id_map(args.gt_dir)
    mask_map = build_id_map(args.mask_dir)

    common_ids = sorted(set(gt_map.keys()) & set(mask_map.keys()), key=lambda x: int(x) if x.isdigit() else x)

    if not common_ids:
        raise RuntimeError("No common image ids found between gt-dir and mask-dir.")

    print("[INFO] num eval images:", len(common_ids))
    print("[INFO] first ids:", common_ids[:10])

    ymls = sorted(yml_dir.glob("*.yml"))
    if not ymls:
        raise RuntimeError(f"No yml files found in {yml_dir}")

    tmp_yml_dir = output_dir / "_tmp_ymls"
    hard_root = output_dir / "images_hard"
    metric_names = ["G_PSNR", "G_SSIM", "H_PSNR", "H_SSIM", "B_PSNR", "B_SSIM", "B_EdgeF1", "LPIPS"]

    method_to_exp = {}

    for yml in ymls:
        method = infer_method_from_yml(yml)
        exp_name = sanitize_name(f"{args.exp_prefix}_{method}")

        method_to_exp[method] = exp_name

        tmp_yml = tmp_yml_dir / f"{exp_name}.yml"
        patch_yml_name(yml, tmp_yml, exp_name)

        result_dir = results_root / exp_name

        if not args.skip_run:
            if args.skip_existing and result_dir.exists():
                print("[SKIP EXISTING]", result_dir)
            else:
                run_test_for_yml(
                    test_script=Path(args.test_script),
                    yml_path=tmp_yml.resolve(),
                    work_dir=work_dir,
                    dry_run=args.dry_run,
                )

        if args.dry_run:
            continue

        pred_dir = find_prediction_dir(results_root, exp_name, set(common_ids))
        pred_map = build_id_map(pred_dir)

        out_method_dir = hard_root / method
        out_method_dir.mkdir(parents=True, exist_ok=True)

        missing = []
        for image_id in common_ids:
            if image_id not in pred_map:
                missing.append(image_id)
                continue

            out_path = out_method_dir / f"{image_id}.png"
            hard_fuse(
                pred_path=pred_map[image_id],
                gt_path=gt_map[image_id],
                mask_path=mask_map[image_id],
                out_path=out_path,
                mask_white_is_known=args.mask_white_is_known,
            )

        print(f"[HARD] method={method}, saved={len(common_ids)-len(missing)}, missing={len(missing)}")
        if missing:
            print("[WARN] missing first ids:", missing[:10])

    if args.dry_run:
        print("[DRY RUN DONE]")
        return

    use_lpips = not args.no_lpips
    lpips_model = make_lpips_model(use_lpips)

    per_image_rows = []

    methods = sorted([p.name for p in hard_root.iterdir() if p.is_dir()])

    for method in methods:
        method_dir = hard_root / method
        pred_map = build_id_map(method_dir)

        for image_id in common_ids:
            if image_id not in pred_map:
                continue

            metrics = eval_one_image(
                pred_path=pred_map[image_id],
                gt_path=gt_map[image_id],
                mask_path=mask_map[image_id],
                eval_size=args.eval_size,
                boundary_kernel=args.boundary_kernel,
                mask_white_is_known=args.mask_white_is_known,
                lpips_model=lpips_model,
            )

            row = {
                "method": method,
                "image_id": image_id,
                "pred_path": str(pred_map[image_id]),
            }
            row.update(metrics)
            per_image_rows.append(row)

    per_image_fields = ["method", "image_id", "pred_path"] + metric_names
    write_csv(output_dir / "per_image_hard_metrics.csv", per_image_rows, per_image_fields)

    summary_rows = summarize(per_image_rows, metric_names)
    summary_fields = ["method", "num_images"] + metric_names
    write_csv(output_dir / "summary_hard_metrics.csv", summary_rows, summary_fields)

    compare_rows = compare_to_reference(summary_rows, args.reference, metric_names)
    compare_fields = ["method", "reference"]
    for m in metric_names:
        compare_fields += [m, f"ref_{m}", f"delta_{m}", f"better_{m}"]
    write_csv(output_dir / f"compare_vs_{args.reference}.csv", compare_rows, compare_fields)

    pixel_rows = pixel_diff_vs_reference(hard_root, methods, common_ids, args.reference)
    write_csv(
        output_dir / f"pixel_diff_vs_{args.reference}.csv",
        pixel_rows,
        ["method", "reference", "image_id", "max_abs_diff", "mean_abs_diff", "pixel_identical"],
    )

    print("\n[DONE]")
    print("Hard images:", hard_root)
    print("Summary CSV:", output_dir / "summary_hard_metrics.csv")
    print("Compare CSV:", output_dir / f"compare_vs_{args.reference}.csv")
    print("Pixel diff CSV:", output_dir / f"pixel_diff_vs_{args.reference}.csv")


if __name__ == "__main__":
    main()
