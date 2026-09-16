import re
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# =========================
# root paths
# =========================
# BASE_SAVE_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/train/results/inpainting/test_muraldh_single")

# OURS_ROOT = BASE_SAVE_ROOT / "muralDH_Test_FusionAblation_Results"
# BASELINE_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/test/results/inpainting/ir-sde/Val_Dataset/new")


BASE_SAVE_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/train/results/inpainting/test_muraldh_ours_final")

OURS_ROOT = BASE_SAVE_ROOT / "MuralDH_Test_FusionAblation_Results"

BASELINE_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/test/results/inpainting/ir-sde_muraldh_baseline_final/MuralDH_Baseline_RouteB_Final")


OUTPUT_ROOT = BASE_SAVE_ROOT / "stitched_compare_heatmap"

OURS_RAW_DIR = OURS_ROOT / "images_raw"
OURS_HARD_DIR = OURS_ROOT / "images_hard"
OURS_SOFT_DIR = OURS_ROOT / "images_soft"

CATEGORIES = ["0-20%", "20-40%", "40%+"]

for fusion in ["hard", "soft"]:
    for cat in CATEGORIES:
        (OUTPUT_ROOT / fusion / cat).mkdir(parents=True, exist_ok=True)


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
        if mode == "hard" and "hardblend" in n:
            return p
        if mode == "soft" and "softblend" in n:
            return p
        if mode == "baseline" and n.endswith("_f.png"):
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


def make_error_heatmap(gt_img: Image.Image, pred_img: Image.Image):
    """
    生成简单误差热力图:
    heat = mean(abs(pred - gt), channel)
    再映射成伪彩色: 黑 -> 蓝 -> 黄 -> 红
    """
    if gt_img is None or pred_img is None:
        return None

    # 统一尺寸
    if gt_img.size != pred_img.size:
        pred_img = pred_img.resize(gt_img.size, Image.BICUBIC)

    gt = np.asarray(gt_img).astype(np.float32)
    pred = np.asarray(pred_img).astype(np.float32)

    diff = np.abs(pred - gt).mean(axis=2) / 255.0  # [H, W], 0~1

    # 拉伸一点对比度，便于看差异
    diff = np.clip(diff * 3.0, 0.0, 1.0)

    # 伪彩色映射
    # R: 高误差强
    # G: 中误差强
    # B: 低误差也保留
    r = np.clip(4 * diff - 1.5, 0, 1)
    g = np.clip(4 * diff - 0.5, 0, 1)
    b = np.clip(1.5 - 4 * diff, 0, 1)

    heat = np.stack([r, g, b], axis=2)
    heat = (heat * 255).astype(np.uint8)

    return Image.fromarray(heat)


def process_category(cat):
    print(f"\n=== Processing category: {cat} ===")

    ours_raw_files = find_all_pngs(OURS_RAW_DIR / cat)
    ours_hard_files = find_all_pngs(OURS_HARD_DIR / cat)
    ours_soft_files = find_all_pngs(OURS_SOFT_DIR / cat)
    baseline_files = find_all_pngs(BASELINE_ROOT / cat)

    raw_index = build_index(ours_raw_files)
    hard_index = build_index(ours_hard_files)
    soft_index = build_index(ours_soft_files)
    baseline_index = build_index(baseline_files)

    all_ids = sorted(set(raw_index.keys()) | set(hard_index.keys()) | set(soft_index.keys()) | set(baseline_index.keys()))

    saved_hard = 0
    saved_soft = 0

    for bid in all_ids:
        gt_path = choose_file(raw_index.get(bid, []), "gt")
        raw_path = choose_file(raw_index.get(bid, []), "raw")
        hard_path = choose_file(hard_index.get(bid, []), "hard")
        soft_path = choose_file(soft_index.get(bid, []), "soft")
        baseline_path = choose_file(baseline_index.get(bid, []), "baseline")

        gt_img = load_rgb(gt_path)
        raw_img = load_rgb(raw_path)
        hard_img = load_rgb(hard_path)
        soft_img = load_rgb(soft_path)
        baseline_img = load_rgb(baseline_path)

        # 第一行：原图对比
        row_hard_top = stitch_horizontal([
            ("GT", gt_img),
            ("Baseline", baseline_img),
            ("Ours Raw", raw_img),
            ("Ours Hard", hard_img),
        ])

        row_soft_top = stitch_horizontal([
            ("GT", gt_img),
            ("Baseline", baseline_img),
            ("Ours Raw", raw_img),
            ("Ours Soft", soft_img),
        ])

        # 第二行：热力图
        baseline_heat = make_error_heatmap(gt_img, baseline_img)
        raw_heat = make_error_heatmap(gt_img, raw_img)
        hard_heat = make_error_heatmap(gt_img, hard_img)
        soft_heat = make_error_heatmap(gt_img, soft_img)

        row_hard_bottom = stitch_horizontal([
            ("Baseline Heatmap", baseline_heat),
            ("Ours Raw Heatmap", raw_heat),
            ("Ours Hard Heatmap", hard_heat),
        ])

        row_soft_bottom = stitch_horizontal([
            ("Baseline Heatmap", baseline_heat),
            ("Ours Raw Heatmap", raw_heat),
            ("Ours Soft Heatmap", soft_heat),
        ])

        # 纵向拼
        final_hard = stitch_vertical([row_hard_top, row_hard_bottom])
        final_soft = stitch_vertical([row_soft_top, row_soft_bottom])

        if final_hard is not None:
            out_hard = OUTPUT_ROOT / "hard" / cat / f"{bid}_compare_heatmap_hard.png"
            final_hard.save(out_hard)
            saved_hard += 1

        if final_soft is not None:
            out_soft = OUTPUT_ROOT / "soft" / cat / f"{bid}_compare_heatmap_soft.png"
            final_soft.save(out_soft)
            saved_soft += 1

    print(f"[OK] {cat}: hard={saved_hard}, soft={saved_soft}")


def main():
    for cat in CATEGORIES:
        process_category(cat)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
