import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# =========================================================
# Paths
# =========================================================
GT_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/train/results/inpainting/test_combined_routeA/DUNHUANG_Test_FusionAblation_Results/images_raw")
BASELINE_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/test/results/inpainting/ir-sde/Val_Dataset/new")
OURS_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/train/results/inpainting/test_combined_routeA/DUNHUANG_Test_FusionAblation_Results/images_soft")
REAL_MASK_ROOT = Path("/root/autodl-tmp/data/mural/DUNHUANG/test/test_mask")

SAVE_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/train/results/inpainting/test_combined_routeA/compare_heatmap_edge_mask")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["0-20%", "20-40%", "40%+"]

for cat in CATEGORIES:
    (SAVE_ROOT / cat).mkdir(parents=True, exist_ok=True)

# =========================================================
# Params
# =========================================================
CANNY_LOW = 80
CANNY_HIGH = 160
BOUNDARY_KERNEL = 5

# 是否只在 hole 区域显示热力图
HEATMAP_ONLY_IN_HOLE = True

# mask overlay color
MASK_COLOR = (255, 0, 0)       # red
BOUNDARY_COLOR = (0, 255, 0)   # green

# =========================================================
# Utils
# =========================================================
def find_all_pngs(root: Path):
    if not root.exists():
        return []
    return list(root.rglob("*.png"))


def extract_base_id(name: str):
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
        if mode == "raw" and n.endswith("_raw.png"):
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


def pil_to_gray_np(img: Image.Image):
    return np.array(img.convert("L"))


def extract_edge(gray_np, low=CANNY_LOW, high=CANNY_HIGH):
    edge = cv2.Canny(gray_np, low, high)
    edge = (edge > 0).astype(np.uint8)
    return edge


def edge_to_rgb(edge_np, color=(255, 255, 255)):
    h, w = edge_np.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[edge_np > 0] = color
    return Image.fromarray(out)


def load_real_mask_by_id(bid, target_hw):
    """
    output:
      mask_known: 1 = known, 0 = hole
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

    if mask_np.shape != target_hw:
        mask_np = cv2.resize(
            mask_np,
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_NEAREST
        )

    mask_bin = (mask_np > 127).astype(np.uint8)  # 1 known, 0 hole
    return mask_bin


def make_boundary_ring(mask_known, kernel_size=BOUNDARY_KERNEL):
    hole = 1 - mask_known
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(hole.astype(np.uint8), kernel, iterations=1)
    ring = np.clip(dilated - hole, 0, 1).astype(np.uint8)
    return ring.astype(np.uint8), hole.astype(np.uint8)


# =========================================================
# Visualization
# =========================================================
def add_title_bar(img, title, bar_h=34):
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


def stitch_vertical(rows, gap=12, bg=(255, 255, 255)):
    valid = [r for r in rows if r is not None]
    if not valid:
        return None

    max_w = max(r.size[0] for r in valid)
    total_h = sum(r.size[1] for r in valid) + gap * (len(valid) - 1)

    canvas = Image.new("RGB", (max_w, total_h), bg)
    y = 0
    for r in valid:
        canvas.paste(r, (0, y))
        y += r.size[1] + gap
    return canvas


def overlay_mask_on_image(img: Image.Image, region_mask, color=(255, 0, 0), alpha=0.35):
    """
    region_mask: 0/1, 1 means highlight
    """
    arr = np.array(img).astype(np.float32)
    overlay = arr.copy()

    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    m = region_mask.astype(bool)

    overlay[m] = overlay[m] * (1 - alpha) + color_arr * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def overlay_boundary_and_hole(img: Image.Image, hole_mask, boundary_mask,
                              hole_color=(255, 0, 0), boundary_color=(0, 255, 0),
                              alpha_hole=0.25, alpha_boundary=0.55):
    arr = np.array(img).astype(np.float32)

    hole_overlay = arr.copy()
    hole_color_arr = np.array(hole_color, dtype=np.float32).reshape(1, 1, 3)
    hole_region = hole_mask.astype(bool)
    hole_overlay[hole_region] = hole_overlay[hole_region] * (1 - alpha_hole) + hole_color_arr * alpha_hole

    boundary_overlay = hole_overlay.copy()
    boundary_color_arr = np.array(boundary_color, dtype=np.float32).reshape(1, 1, 3)
    boundary_region = boundary_mask.astype(bool)
    boundary_overlay[boundary_region] = boundary_overlay[boundary_region] * (1 - alpha_boundary) + boundary_color_arr * alpha_boundary

    boundary_overlay = np.clip(boundary_overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(boundary_overlay)


def make_error_heatmap(gt_img: Image.Image, pred_img: Image.Image, region_mask=None):
    """
    error heatmap:
      heat = mean(abs(pred - gt), channel)
    若给 region_mask，则只在 region_mask 内显示热力图，其余区域保留原图或变暗。
    """
    if gt_img is None or pred_img is None:
        return None

    if gt_img.size != pred_img.size:
        pred_img = pred_img.resize(gt_img.size, Image.BICUBIC)

    gt = np.asarray(gt_img).astype(np.float32)
    pred = np.asarray(pred_img).astype(np.float32)

    diff = np.abs(pred - gt).mean(axis=2) / 255.0
    diff = np.clip(diff * 3.0, 0.0, 1.0)

    r = np.clip(4 * diff - 1.5, 0, 1)
    g = np.clip(4 * diff - 0.5, 0, 1)
    b = np.clip(1.5 - 4 * diff, 0, 1)
    heat = np.stack([r, g, b], axis=2)
    heat = (heat * 255).astype(np.uint8)

    if region_mask is None:
        return Image.fromarray(heat)

    region_mask = region_mask.astype(bool)

    # 背景用 GT 变暗，hole 区域显示热力图
    bg = (gt * 0.45).astype(np.uint8)
    out = bg.copy()
    out[region_mask] = heat[region_mask]
    return Image.fromarray(out)


# =========================================================
# Main
# =========================================================
def process_category(cat):
    print(f"\n=== Processing category: {cat} ===")

    gt_files = find_all_pngs(GT_ROOT / cat)
    baseline_files = find_all_pngs(BASELINE_ROOT / cat)
    ours_files = find_all_pngs(OURS_ROOT / cat)

    gt_index = build_index(gt_files)
    baseline_index = build_index(baseline_files)
    ours_index = build_index(ours_files)

    all_ids = sorted(set(gt_index.keys()) & set(baseline_index.keys()) & set(ours_index.keys()))
    print(f"[{cat}] matched ids = {len(all_ids)}")

    saved = 0

    for bid in all_ids:
        gt_path = choose_file(gt_index.get(bid, []), "gt")
        baseline_path = choose_file(baseline_index.get(bid, []), "baseline")
        ours_path = choose_file(ours_index.get(bid, []), "ours_soft")

        gt_img = load_rgb(gt_path)
        baseline_img = load_rgb(baseline_path)
        ours_img = load_rgb(ours_path)

        if gt_img is None or baseline_img is None or ours_img is None:
            continue

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
            print(f"  [skip] no mask for {bid}")
            continue

        boundary_ring, hole_region = make_boundary_ring(mask_known)

        gt_mask_overlay = overlay_mask_on_image(gt_img, hole_region, color=MASK_COLOR, alpha=0.35)
        gt_boundary_overlay = overlay_boundary_and_hole(
            gt_img,
            hole_region,
            boundary_ring,
            hole_color=MASK_COLOR,
            boundary_color=BOUNDARY_COLOR,
            alpha_hole=0.20,
            alpha_boundary=0.55
        )

        if HEATMAP_ONLY_IN_HOLE:
            baseline_heat = make_error_heatmap(gt_img, baseline_img, region_mask=hole_region)
            ours_heat = make_error_heatmap(gt_img, ours_img, region_mask=hole_region)
        else:
            baseline_heat = make_error_heatmap(gt_img, baseline_img)
            ours_heat = make_error_heatmap(gt_img, ours_img)

        row1 = stitch_horizontal([
            ("GT", gt_img),
            ("GT + Hole Mask", gt_mask_overlay),
            ("GT + Boundary Ring", gt_boundary_overlay),
            ("Baseline", baseline_img),
            ("Ours", ours_img),
        ])

        row2 = stitch_horizontal([
            ("Baseline Heatmap", baseline_heat),
            ("Ours Heatmap", ours_heat),
            ("GT Edge", edge_to_rgb(gt_edge)),
            ("Baseline Edge", edge_to_rgb(baseline_edge)),
            ("Ours Edge", edge_to_rgb(ours_edge)),
        ])

        final = stitch_vertical([row1, row2])

        if final is not None:
            out_path = SAVE_ROOT / cat / f"{bid}_compare_mask_heatmap_edge.png"
            final.save(out_path)
            saved += 1

    print(f"[OK] {cat}: saved={saved}")


def main():
    for cat in CATEGORIES:
        process_category(cat)

    print("\nDone.")
    print("Saved to:", SAVE_ROOT)


if __name__ == "__main__":
    main()
