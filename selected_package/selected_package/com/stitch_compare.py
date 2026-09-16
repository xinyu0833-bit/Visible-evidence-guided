import os
import re
from pathlib import Path
from PIL import Image, ImageDraw

# =========================
# root paths
# =========================
# BASE_SAVE_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/train/results/inpainting/test_muraldh_single")

# OURS_ROOT = BASE_SAVE_ROOT / "muralDH_Test_FusionAblation_Results"
# BASELINE_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/test/results/inpainting/ir-sde/Val_Dataset/new")

BASE_SAVE_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/train/results/inpainting/test_muraldh_ours_final")

OURS_ROOT = BASE_SAVE_ROOT / "MuralDH_Test_FusionAblation_Results"

BASELINE_ROOT = Path("/root/autodl-tmp/StrDiffusion_Official/test/results/inpainting/ir-sde_muraldh_baseline/MuralDH_Baseline_RouteB_Final")


OUTPUT_ROOT = BASE_SAVE_ROOT / "stitched_compare"

# ours subdirs
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
    """
    提取样本主 id
    例如:
      502_raw.png -> 502
      502_r.png -> 502
      502_0-20_ratio0.18_HardBlend.png -> 502
      502_0-20_ratio0.18_SoftBlend.png -> 502
      502_f.png -> 502
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
    """
    mode:
      gt / raw / hard / soft / baseline
    """
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
    """
    items: list of (title, image_or_none)
    """
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

        # hard stitched
        stitched_hard = stitch_horizontal([
            ("GT", gt_img),
            ("Baseline", baseline_img),
            ("Ours Raw", raw_img),
            ("Ours Hard", hard_img),
        ])

        if stitched_hard is not None:
            out_hard = OUTPUT_ROOT / "hard" / cat / f"{bid}_compare_hard.png"
            stitched_hard.save(out_hard)
            saved_hard += 1

        # soft stitched
        stitched_soft = stitch_horizontal([
            ("GT", gt_img),
            ("Baseline", baseline_img),
            ("Ours Raw", raw_img),
            ("Ours Soft", soft_img),
        ])

        if stitched_soft is not None:
            out_soft = OUTPUT_ROOT / "soft" / cat / f"{bid}_compare_soft.png"
            stitched_soft.save(out_soft)
            saved_soft += 1

    print(f"[OK] {cat}: hard={saved_hard}, soft={saved_soft}")


def main():
    for cat in CATEGORIES:
        process_category(cat)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
