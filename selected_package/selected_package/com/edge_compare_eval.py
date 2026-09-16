import os
import re
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# =========================
# paths
# =========================
# # GT 从 ours 的 raw 目录里拿 *_r.png
GT_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/train/results/inpainting/test_combined_routeA/DUNHUANG_Test_FusionAblation_Results/images_raw")

# Baseline 结果目录（按档位分）
BASELINE_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/test/results/inpainting/ir-sde/Val_Dataset/new")

# Ours 结果目录（可改成 images_hard 或 images_soft）
OURS_RESULT_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/train/results/inpainting/test_combined_routeA/DUNHUANG_Test_FusionAblation_Results/images_soft")

# 真实 mask 根目录
REAL_MASK_ROOT = Path("/root/autodl-tmp/data/mural/DUNHUANG/test/test_mask")

SAVE_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/train/results/inpainting/test_combined_routeA/edge_compare_results_realmask")


# GT_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/train/results/inpainting/test_muraldh_ours_final/MuralDH_Test_FusionAblation_Results/images_raw")

# BASELINE_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/test/results/inpainting/ir-sde_muraldh_baseline_final/MuralDH_Baseline_RouteB_Final")

# OURS_RESULT_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/train/results/inpainting/test_muraldh_ours_final/MuralDH_Test_FusionAblation_Results/images_soft")
# # 如果你要比较 hard，就改成 images_hard

# REAL_MASK_ROOT = Path("/root/autodl-tmp/data/mural/muralDH/test_mask")

# SAVE_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/train/results/inpainting/test_muraldh_ours_final/edge_compare_results_realmask")


VIS_ROOT = SAVE_ROOT / "visualizations"
CSV_PATH = SAVE_ROOT / "edge_metrics_realmask.csv"

CATEGORIES = ["0-20%", "20-40%", "40%+"]

for cat in CATEGORIES:
    (VIS_ROOT / cat).mkdir(parents=True, exist_ok=True)
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

# Canny params
CANNY_LOW = 80
CANNY_HIGH = 160

# boundary ring dilation kernel
BOUNDARY_KERNEL = 5


def find_all_pngs(root: Path):
    if not root.exists():
        return []
    return list(root.rglob("*.png"))


def extract_base_id(name: str):
    """
    例如:
      502_r.png -> 502
      502_f.png -> 502
      502_0-20_ratio0.18_SoftBlend.png -> 502
    """
    stem = Path(name).stem
    m = re.match(r"^([A-Za-z0-9]+)", stem)
    if m:
        return m.group(1)
    return stem


def build_index(file_list):
    out = {}
    for p in file_list:
        bid = extract_base_id(p.name)
        out.setdefault(bid, []).append(p)
    return out


def choose_file(paths, mode):
    if not paths:
        return None

    for p in paths:
        n = p.name.lower()
        if mode == "gt" and n.endswith("_r.png"):
            return p
        if mode == "baseline" and n.endswith("_f.png"):
            return p
        if mode == "ours_soft" and "softblend" in n:
            return p
        if mode == "ours_hard" and "hardblend" in n:
            return p

    if mode == "baseline":
        for p in paths:
            if p.name.lower().endswith(".png"):
                return p
    return None


def load_rgb(path):
    if path is None or not path.exists():
        return None
    return Image.open(path).convert("RGB")


def pil_to_gray_np(img):
    return np.array(img.convert("L"))


def extract_edge(gray_np, low=CANNY_LOW, high=CANNY_HIGH):
    edge = cv2.Canny(gray_np, low, high)
    edge = (edge > 0).astype(np.uint8)
    return edge


def load_real_mask_by_id(bid, target_hw):
    """
    读取真实 mask:
    约定输出为 mask_known:
      1 = known region
      0 = hole region
    """
    candidates = [
        REAL_MASK_ROOT / f"{bid}.png",
        REAL_MASK_ROOT / f"{bid}.jpg",
        REAL_MASK_ROOT / f"{bid}_mask.png",
        REAL_MASK_ROOT / f"{bid}_mask.jpg",
        REAL_MASK_ROOT / f"{bid}_masked.png",
        REAL_MASK_ROOT / f"{bid}_masked.jpg",
    ]

    mask_path = None
    for p in candidates:
        if p.exists():
            mask_path = p
            break

    if mask_path is None:
        return None

    mask_img = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_img)

    # resize to GT size
    if mask_np.shape != target_hw:
        mask_np = cv2.resize(mask_np, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)

    # 二值化
    mask_bin = (mask_np > 127).astype(np.uint8)

    # 约定：1=known, 0=hole
    return mask_bin


def make_boundary_ring(mask_known, kernel_size=BOUNDARY_KERNEL):
    """
    mask_known: 1=known, 0=hole
    """
    hole = 1 - mask_known
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(hole.astype(np.uint8), kernel, iterations=1)
    ring = np.clip(dilated - hole, 0, 1).astype(np.uint8)
    return ring, hole.astype(np.uint8)


def compute_prf(pred, gt, region_mask=None):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    if region_mask is not None:
        m = region_mask.astype(bool)
        pred = pred[m]
        gt = gt[m]

    # 如果该区域没有像素，返回 NaN，避免误导成 0
    if pred.size == 0:
        return np.nan, np.nan, np.nan

    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return float(precision), float(recall), float(f1)


def edge_to_rgb(edge_np, color=(255, 255, 255)):
    h, w = edge_np.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[edge_np > 0] = color
    return Image.fromarray(out)


def add_title_bar(img, title, bar_h=32):
    w, h = img.size
    canvas = Image.new("RGB", (w, h + bar_h), (255, 255, 255))
    canvas.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(0, 0, 0))
    return canvas


def resize_to_same_height(imgs):
    valid = [im for im in imgs if im is not None]
    if not valid:
        return imgs

    target_h = min(im.size[1] for im in valid)
    out = []
    for im in imgs:
        if im is None:
            out.append(None)
            continue
        w, h = im.size
        new_w = int(w * target_h / h)
        out.append(im.resize((new_w, target_h), Image.BICUBIC))
    return out


def stitch_horizontal(items, gap=10, bg=(240, 240, 240)):
    valid = [(t, im) for t, im in items if im is not None]
    if not valid:
        return None

    titles = [t for t, _ in valid]
    imgs = [im for _, im in valid]
    imgs = resize_to_same_height(imgs)
    imgs = [add_title_bar(im, t) for im, t in zip(imgs, titles)]

    total_w = sum(im.size[0] for im in imgs) + gap * (len(imgs) - 1)
    max_h = max(im.size[1] for im in imgs)

    canvas = Image.new("RGB", (total_w, max_h), bg)
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.size[0] + gap
    return canvas


def safe_mean(vals):
    vals = [v for v in vals if not np.isnan(v)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def main():
    gt_index = build_index(find_all_pngs(GT_ROOT))
    baseline_index = build_index(find_all_pngs(BASELINE_ROOT))
    ours_index = build_index(find_all_pngs(OURS_RESULT_ROOT))

    rows = []

    ours_mode = "ours_soft" if "images_soft" in str(OURS_RESULT_ROOT) else "ours_hard"

    for cat in CATEGORIES:
        gt_cat_files = find_all_pngs(GT_ROOT / cat)
        baseline_cat_files = find_all_pngs(BASELINE_ROOT / cat)
        ours_cat_files = find_all_pngs(OURS_RESULT_ROOT / cat)

        gt_cat_index = build_index(gt_cat_files)
        baseline_cat_index = build_index(baseline_cat_files)
        ours_cat_index = build_index(ours_cat_files)

        ids = sorted(
            set(gt_cat_index.keys()) &
            set(baseline_cat_index.keys()) &
            set(ours_cat_index.keys())
        )

        print(f"\n[{cat}] matched ids = {len(ids)}")

        for bid in ids:
            gt_path = choose_file(gt_cat_index.get(bid, []), "gt")
            baseline_path = choose_file(baseline_cat_index.get(bid, []), "baseline")
            ours_path = choose_file(ours_cat_index.get(bid, []), ours_mode)

            gt_img = load_rgb(gt_path)
            baseline_img = load_rgb(baseline_path)
            ours_img = load_rgb(ours_path)

            if gt_img is None or baseline_img is None or ours_img is None:
                continue

            # resize all to GT size
            baseline_img = baseline_img.resize(gt_img.size, Image.BICUBIC)
            ours_img = ours_img.resize(gt_img.size, Image.BICUBIC)

            gt_gray = pil_to_gray_np(gt_img)
            baseline_gray = pil_to_gray_np(baseline_img)
            ours_gray = pil_to_gray_np(ours_img)

            gt_edge = extract_edge(gt_gray)
            baseline_edge = extract_edge(baseline_gray)
            ours_edge = extract_edge(ours_gray)

            mask_known = load_real_mask_by_id(bid, gt_edge.shape)
            if mask_known is None:
                print(f"  [skip] no real mask for {bid}")
                continue

            boundary_ring, hole_region = make_boundary_ring(mask_known)

            # full
            b_p, b_r, b_f1 = compute_prf(baseline_edge, gt_edge)
            o_p, o_r, o_f1 = compute_prf(ours_edge, gt_edge)

            # hole
            b_hp, b_hr, b_hf1 = compute_prf(baseline_edge, gt_edge, hole_region)
            o_hp, o_hr, o_hf1 = compute_prf(ours_edge, gt_edge, hole_region)

            # boundary
            b_bp, b_br, b_bf1 = compute_prf(baseline_edge, gt_edge, boundary_ring)
            o_bp, o_br, o_bf1 = compute_prf(ours_edge, gt_edge, boundary_ring)

            rows.append({
                "Image": bid,
                "Category": cat,

                "Baseline_Precision": round(b_p, 4),
                "Baseline_Recall": round(b_r, 4),
                "Baseline_F1": round(b_f1, 4),

                "Ours_Precision": round(o_p, 4),
                "Ours_Recall": round(o_r, 4),
                "Ours_F1": round(o_f1, 4),

                "Baseline_Hole_Precision": round(b_hp, 4) if not np.isnan(b_hp) else "",
                "Baseline_Hole_Recall": round(b_hr, 4) if not np.isnan(b_hr) else "",
                "Baseline_Hole_F1": round(b_hf1, 4) if not np.isnan(b_hf1) else "",

                "Ours_Hole_Precision": round(o_hp, 4) if not np.isnan(o_hp) else "",
                "Ours_Hole_Recall": round(o_hr, 4) if not np.isnan(o_hr) else "",
                "Ours_Hole_F1": round(o_hf1, 4) if not np.isnan(o_hf1) else "",

                "Baseline_Boundary_Precision": round(b_bp, 4) if not np.isnan(b_bp) else "",
                "Baseline_Boundary_Recall": round(b_br, 4) if not np.isnan(b_br) else "",
                "Baseline_Boundary_F1": round(b_bf1, 4) if not np.isnan(b_bf1) else "",

                "Ours_Boundary_Precision": round(o_bp, 4) if not np.isnan(o_bp) else "",
                "Ours_Boundary_Recall": round(o_br, 4) if not np.isnan(o_br) else "",
                "Ours_Boundary_F1": round(o_bf1, 4) if not np.isnan(o_bf1) else "",
            })

            vis = stitch_horizontal([
                ("GT", gt_img),
                ("GT Edge", edge_to_rgb(gt_edge)),
                ("Baseline", baseline_img),
                ("Baseline Edge", edge_to_rgb(baseline_edge)),
                ("Ours", ours_img),
                ("Ours Edge", edge_to_rgb(ours_edge)),
            ])

            if vis is not None:
                vis.save(VIS_ROOT / cat / f"{bid}_edge_compare.png")

    if len(rows) == 0:
        print("No matched samples found.")
        return

    # save CSV
    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # summary by category
    print("\n================ Edge Comparison Summary (Real Mask) ================")
    for cat in CATEGORIES:
        sub = [r for r in rows if r["Category"] == cat]
        if not sub:
            continue

        def collect(key):
            vals = []
            for r in sub:
                v = r[key]
                if v != "":
                    vals.append(float(v))
            return vals

        print(f"\n[{cat}]  count = {len(sub)}")
        print("Baseline Full F1     :", round(safe_mean(collect("Baseline_F1")), 4))
        print("Ours Full F1         :", round(safe_mean(collect("Ours_F1")), 4))
        print("Baseline Hole F1     :", round(safe_mean(collect("Baseline_Hole_F1")), 4))
        print("Ours Hole F1         :", round(safe_mean(collect("Ours_Hole_F1")), 4))
        print("Baseline Boundary F1 :", round(safe_mean(collect("Baseline_Boundary_F1")), 4))
        print("Ours Boundary F1     :", round(safe_mean(collect("Ours_Boundary_F1")), 4))

    print("\nSaved CSV:", CSV_PATH)
    print("Saved visualizations:", VIS_ROOT)


if __name__ == "__main__":
    main()
