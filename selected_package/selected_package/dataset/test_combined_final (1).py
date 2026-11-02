#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_combined_final.py

Prompt-ablation ready test script for StrDiffusion Dunhuang inpainting.

Main fixes compared with the previous test script:
1. Accepts prompt-ablation CLI arguments:
   --prompt_json, --prompt_mode, --ablation_name, --max_images,
   --disable_prompt_adapter, --route_mode, --prompt_injection,
   --adapter_variant, --prompt_scale
2. Uses prompt_json + prompt_mode to select the correct prompt for each image.
3. Saves outputs under per-ablation folders, so different prompt tests do not overwrite each other.
4. Keeps quick runtime metrics only for monitoring. Paper metrics should be computed by
   evaluate_prompt_ablation_metrics_paper.py after all images are generated.

Expected mask rule:
    mask = 1 means known / valid region
    mask = 0 means hole / missing region
"""

import argparse
import glob
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.feature import canny
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import gaussian_blur
from transformers import CLIPTextModel, CLIPTokenizer

import options as option
import str_utils as str_util  # keep original project import
import utils as util
from data import create_dataloader, create_dataset
from models import create_model

from ablation_prompt_utils import (
    add_ablation_arguments,
    apply_ablation_to_model,
    load_prompt_database,
    select_prompt,
)

try:
    import lpips
    HAS_LPIPS = True
except Exception:
    lpips = None
    HAS_LPIPS = False

try:
    import pyiqa
    HAS_NIQE = True
except Exception:
    pyiqa = None
    HAS_NIQE = False


class OutputWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.current_mask = None
        self.S_GT = None
        self.S_LQ = None
        self.S_sde = None

    def forward(self, xt, cond, time, **kwargs):
        if self.current_mask is not None:
            kwargs["mask"] = self.current_mask

        if self.S_sde is not None and self.S_GT is not None and self.S_LQ is not None:
            if isinstance(time, (int, float)):
                t_tensor = torch.tensor([int(time)], device=xt.device).long()
            else:
                t_tensor = time.long().squeeze()
                if t_tensor.dim() == 0:
                    t_tensor = t_tensor.unsqueeze(0)

            self.S_sde.set_mu(self.S_LQ * self.current_mask)
            S_mu_bar = self.S_sde.mu_bar(self.S_GT, t_tensor)
            S_sigma_bar = self.S_sde.sigma_bar(t_tensor).view(-1, 1, 1, 1)
            S_noise = torch.randn_like(S_mu_bar)
            S_optimum = S_mu_bar + S_noise * S_sigma_bar
            kwargs["S"] = S_optimum

        out = self.net(xt, cond, time, **kwargs)
        if isinstance(out, tuple):
            return out[0]
        return out


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_ssim_rgb(a_rgb_f32, b_rgb_f32):
    """Version-tolerant RGB SSIM for quick runtime monitoring."""
    try:
        return float(ssim_metric(
            a_rgb_f32,
            b_rgb_f32,
            channel_axis=2,
            data_range=1.0
        ))
    except TypeError:
        return float(ssim_metric(
            a_rgb_f32,
            b_rgb_f32,
            multichannel=True,
            data_range=1.0
        ))


def compute_quick_metrics(pred_tensor, gt_tensor, mask_tensor, loss_fn_vgg=None, niqe_metric=None):
    """
    Quick monitoring metrics only. Do not use this table directly for the final paper.
    For paper tables, run evaluate_prompt_ablation_metrics_paper.py on saved outputs.
    """
    pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)
    gt_tensor = torch.clamp(gt_tensor, 0.0, 1.0)

    pred_np = pred_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    gt_np = gt_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    mask_np = mask_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()

    pred_u8 = np.clip(pred_np * 255.0, 0, 255).astype(np.uint8)
    gt_u8 = np.clip(gt_np * 255.0, 0, 255).astype(np.uint8)

    pred_f32 = pred_u8.astype(np.float32) / 255.0
    gt_f32 = gt_u8.astype(np.float32) / 255.0

    global_psnr = psnr_metric(gt_u8, pred_u8, data_range=255)
    global_ssim = safe_ssim_rgb(gt_f32, pred_f32)

    hole_1c = 1.0 - mask_np
    hole_3c = np.expand_dims(hole_1c, axis=-1).repeat(3, axis=-1)

    mse_hole = np.sum(((gt_f32 - pred_f32) ** 2) * hole_3c) / (np.sum(hole_3c) + 1e-8)
    hole_psnr = 10.0 * np.log10(1.0 / mse_hole) if mse_hole > 0 else 99.0

    lpips_val = np.nan
    if loss_fn_vgg is not None:
        with torch.no_grad():
            t_pred = pred_tensor * 2.0 - 1.0
            t_gt = gt_tensor * 2.0 - 1.0
            lpips_val = float(loss_fn_vgg(t_pred, t_gt).item())

    niqe_val = np.nan
    if niqe_metric is not None:
        try:
            with torch.no_grad():
                niqe_val = float(niqe_metric(pred_tensor).item())
        except Exception:
            niqe_val = np.nan

    return {
        "Global_PSNR": round(float(global_psnr), 4),
        "Global_SSIM": round(float(global_ssim), 4),
        "Hole_PSNR": round(float(hole_psnr), 4),
        "LPIPS": round(float(lpips_val), 4) if lpips_val == lpips_val else np.nan,
        "NIQE": round(float(niqe_val), 4) if niqe_val == niqe_val else np.nan,
    }


def normalize_img_id(name):
    stem = Path(str(name)).stem
    for suffix in ["_masked", "_mask", "_gt", "_GT"]:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    return stem


def find_mask_path(mask_root, img_name, available_masks):
    image_id = normalize_img_id(img_name)
    possible_mask_names = [
        f"{img_name}.jpg", f"{img_name}.png",
        f"{image_id}.jpg", f"{image_id}.png",
        f"{img_name}_mask.jpg", f"{img_name}_mask.png",
        f"{image_id}_mask.jpg", f"{image_id}_mask.png",
        f"{img_name}_masked.jpg", f"{img_name}_masked.png",
        f"{image_id}_masked.jpg", f"{image_id}_masked.png",
    ]

    for name in possible_mask_names:
        tmp_path = os.path.join(mask_root, name)
        if os.path.exists(tmp_path):
            return tmp_path

    # Last fallback: match first digit group.
    import re
    m = re.search(r"(\d+)", image_id)
    if m:
        digits = m.group(1)
        for p in available_masks:
            st = Path(p).stem
            if st == digits or st.startswith(digits + "_") or digits in st:
                return p

    return None


def load_legacy_prompt_dict(dataset_opt):
    prompt_dict = {}
    prompt_json_path = dataset_opt.get("prompt_path", "")
    if prompt_json_path and os.path.exists(prompt_json_path):
        with open(prompt_json_path, "r", encoding="utf-8") as f:
            prompt_dict = json.load(f)
    return prompt_dict


def select_legacy_prompt(prompt_dict, img_name):
    raw_prompt = prompt_dict.get(img_name, None)
    if raw_prompt is None:
        raw_prompt = prompt_dict.get(normalize_img_id(img_name), "A Dunhuang mural fragment with mineral pigments.")
    if isinstance(raw_prompt, dict):
        return str(raw_prompt.get("final_prompt") or raw_prompt.get("prompt") or raw_prompt.get("text") or "A Dunhuang mural fragment with mineral pigments.")
    return str(raw_prompt)


def summarize_results(df, title):
    print("\n" + "=" * 110)
    print(title)
    print(f"{'Category':<12} | {'Count':<6} | {'G-PSNR':<8} | {'H-PSNR':<8} | {'LPIPS':<8} | {'NIQE':<8}")
    print("-" * 110)

    categories = ["0-20%", "20-40%", "40%+"]
    summary_rows = []

    for c in categories:
        sub = df[df["Category"] == c]
        if len(sub) == 0:
            continue

        m_gp = pd.to_numeric(sub["Global_PSNR"], errors="coerce").mean()
        m_hp = pd.to_numeric(sub["Hole_PSNR"], errors="coerce").mean()
        m_lp = pd.to_numeric(sub["LPIPS"], errors="coerce").mean()
        m_nq = pd.to_numeric(sub["NIQE"], errors="coerce").mean()

        print(f"{c:<12} | {len(sub):<6} | {m_gp:<8.4f} | {m_hp:<8.4f} | {m_lp:<8.4f} | {m_nq:<8.4f}")

        summary_rows.append({
            "Image": f"AVG_{c}",
            "Category": c,
            "Ratio": "",
            "Fusion": title,
            "Global_PSNR": m_gp,
            "Global_SSIM": pd.to_numeric(sub["Global_SSIM"], errors="coerce").mean(),
            "Hole_PSNR": m_hp,
            "LPIPS": m_lp,
            "NIQE": m_nq,
        })

    summary_rows.append({
        "Image": "AVERAGE_ALL",
        "Category": "ALL",
        "Ratio": "",
        "Fusion": title,
        "Global_PSNR": pd.to_numeric(df["Global_PSNR"], errors="coerce").mean(),
        "Global_SSIM": pd.to_numeric(df["Global_SSIM"], errors="coerce").mean(),
        "Hole_PSNR": pd.to_numeric(df["Hole_PSNR"], errors="coerce").mean(),
        "LPIPS": pd.to_numeric(df["LPIPS"], errors="coerce").mean(),
        "NIQE": pd.to_numeric(df["NIQE"], errors="coerce").mean(),
    })

    return summary_rows


def main():
    set_random_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", default="options/test/test_combined_routeA.yml", type=str)
    add_ablation_arguments(parser)
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    base_results_root = opt["path"].get("results_root", os.path.join(os.getcwd(), "results", opt.get("name", "test")))
    util.mkdirs(
        path for key, path in opt["path"].items()
        if key != "experiments_root" and "pretrain_model" not in key and "resume" not in key
    )

    if opt["path"].get("log"):
        util.setup_logger(
            "base",
            opt["path"]["log"],
            f"test_prompt_ablation_{args.ablation_name}",
            level=logging.INFO,
            screen=True,
            tofile=True,
        )
        logger = logging.getLogger("base")
    else:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model...")
    model = create_model(opt)
    apply_ablation_to_model(model, args)

    logger.info("Loading CLIP text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    text_encoder.eval()

    logger.info("Loading quick metric models...")
    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device).eval() if HAS_LPIPS else None
    niqe_metric = pyiqa.create_metric("niqe", device=device) if HAS_NIQE else None

    prompt_db = load_prompt_database(args.prompt_json) if getattr(args, "prompt_json", None) else {}

    sde = util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=device,
    )
    S_sde = util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=device,
    )

    wrapped_model = OutputWrapper(model.model)
    sde.set_model(wrapped_model)

    resize_256 = transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC)
    mask_resize_256 = transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST)
    transform_mask = transforms.Compose([mask_resize_256, transforms.ToTensor()])

    logger.info(f"Ablation name : {args.ablation_name}")
    logger.info(f"Prompt mode   : {args.prompt_mode}")
    logger.info(f"Prompt json   : {args.prompt_json}")
    logger.info(f"Max images    : {args.max_images}")

    for phase, dataset_opt in sorted(opt["datasets"].items()):
        dataset_name = dataset_opt.get("name", phase)
        logger.info("\n" + "=" * 85)
        logger.info(f"Start evaluation dataset: {dataset_name}")
        logger.info("=" * 85)

        current_results_root = os.path.join(base_results_root, f"{dataset_name}_PromptAblation_Results")
        hard_dir = os.path.join(current_results_root, "images_hard", args.ablation_name)
        soft_dir = os.path.join(current_results_root, "images_soft", args.ablation_name)
        raw_dir = os.path.join(current_results_root, "images_raw", args.ablation_name)
        csv_path = os.path.join(current_results_root, f"{dataset_name}_{args.ablation_name}_quick_metrics.csv")

        util.mkdir(current_results_root)
        util.mkdir(hard_dir)
        util.mkdir(soft_dir)
        util.mkdir(raw_dir)

        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        legacy_prompt_dict = load_legacy_prompt_dict(dataset_opt)

        mask_root = dataset_opt.get("mask_root", opt.get("degradation", {}).get("mask_root", ""))
        available_masks = sorted(
            glob.glob(os.path.join(mask_root, "*.png")) +
            glob.glob(os.path.join(mask_root, "*.jpg")) +
            glob.glob(os.path.join(mask_root, "*.jpeg"))
        )

        all_results = []
        evaluated = 0

        for g, train_data in enumerate(test_loader):
            if args.max_images is not None and args.max_images > 0 and evaluated >= args.max_images:
                break

            img_path = train_data["GT_path"][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            image_id = normalize_img_id(img_name)

            mask_path = find_mask_path(mask_root, img_name, available_masks)
            if mask_path is None:
                logger.warning(f"Skip {img_name}: no mask found in {mask_root}")
                continue

            real_gt_pil = Image.open(img_path).convert("RGB")
            Y_GT = transforms.ToTensor()(resize_256(real_gt_pil)).unsqueeze(0).to(device)
            X_GT = transforms.ToTensor()(resize_256(real_gt_pil.convert("L"))).unsqueeze(0).to(device)

            real_gt_gray_np = np.array(resize_256(real_gt_pil.convert("L"))) / 255.0
            edge_np = canny(real_gt_gray_np, sigma=2.0)
            X_LQ = torch.from_numpy(edge_np).float().unsqueeze(0).unsqueeze(0).to(device)

            mask_pil = Image.open(mask_path).convert("1")
            mask = transform_mask(mask_pil).unsqueeze(0).to(device).float()
            mask = (mask > 0.5).float()

            damage_ratio = (mask == 0).float().mean().item()
            cat = "0-20%" if damage_ratio <= 0.20 else "20-40%" if damage_ratio <= 0.40 else "40%+"

            save_hard_subdir = os.path.join(hard_dir, cat)
            save_soft_subdir = os.path.join(soft_dir, cat)
            save_raw_subdir = os.path.join(raw_dir, cat)
            util.mkdir(save_hard_subdir)
            util.mkdir(save_soft_subdir)
            util.mkdir(save_raw_subdir)

            if prompt_db:
                current_prompt = select_prompt(
                    prompt_db=prompt_db,
                    image_id=image_id,
                    prompt_mode=args.prompt_mode,
                    index=g,
                )
            else:
                current_prompt = select_legacy_prompt(legacy_prompt_dict, image_id)

            tokens = tokenizer(
                [current_prompt],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                prompt_context = text_encoder(tokens.input_ids)[0]

            model.model.eval()
            sde.set_mu(Y_GT * mask)

            wrapped_model.current_mask = mask
            wrapped_model.S_GT = X_GT
            wrapped_model.S_LQ = X_LQ
            wrapped_model.S_sde = S_sde

            init_xt = (Y_GT * mask) + torch.randn_like(Y_GT) * (opt["sde"]["max_sigma"] / 255.0)

            with torch.no_grad():
                SR_img = sde.reverse_sde(init_xt, prompt_context=prompt_context)

            wrapped_model.current_mask = None
            wrapped_model.S_GT = None
            wrapped_model.S_LQ = None
            wrapped_model.S_sde = None

            SR_img = torch.clamp(SR_img.to(device), 0.0, 1.0)

            hard_out = SR_img * (1.0 - mask) + Y_GT * mask
            hard_out = torch.clamp(hard_out, 0.0, 1.0)

            soft_mask = gaussian_blur(mask, kernel_size=[5, 5], sigma=[1.5, 1.5])
            soft_out = SR_img * (1.0 - soft_mask) + Y_GT * soft_mask
            soft_out = torch.clamp(soft_out, 0.0, 1.0)

            hard_metrics = compute_quick_metrics(hard_out, Y_GT, mask, loss_fn_vgg, niqe_metric)
            soft_metrics = compute_quick_metrics(soft_out, Y_GT, mask, loss_fn_vgg, niqe_metric)

            all_results.append({
                "Image": image_id,
                "Category": cat,
                "Ratio": round(damage_ratio, 6),
                "Ablation": args.ablation_name,
                "PromptMode": args.prompt_mode,
                "Fusion": "HardBlend",
                "Prompt": current_prompt,
                **hard_metrics,
            })
            all_results.append({
                "Image": image_id,
                "Category": cat,
                "Ratio": round(damage_ratio, 6),
                "Ablation": args.ablation_name,
                "PromptMode": args.prompt_mode,
                "Fusion": "SoftBlend",
                "Prompt": current_prompt,
                **soft_metrics,
            })

            hard_path = os.path.join(
                save_hard_subdir,
                f"{image_id}_{cat.replace('%', '')}_ratio{damage_ratio:.2f}_HardBlend.png",
            )
            soft_path = os.path.join(
                save_soft_subdir,
                f"{image_id}_{cat.replace('%', '')}_ratio{damage_ratio:.2f}_SoftBlend.png",
            )
            raw_path = os.path.join(save_raw_subdir, f"{image_id}_raw.png")
            gt_copy_path = os.path.join(save_raw_subdir, f"{image_id}_gtcopy.png")

            util.save_img(util.tensor2img(hard_out.squeeze(0)), hard_path)
            util.save_img(util.tensor2img(soft_out.squeeze(0)), soft_path)
            util.save_img(util.tensor2img(SR_img.squeeze(0)), raw_path)
            util.save_img(util.tensor2img(Y_GT.squeeze(0)), gt_copy_path)

            evaluated += 1
            if evaluated % 10 == 0:
                logger.info(
                    f"[{dataset_name}] {args.ablation_name}: finished {evaluated} images | "
                    f"Hard G-PSNR={hard_metrics['Global_PSNR']:.2f} | "
                    f"Hard H-PSNR={hard_metrics['Hole_PSNR']:.2f}"
                )

        if len(all_results) > 0:
            df_all = pd.DataFrame(all_results)
            hard_summary = summarize_results(df_all[df_all["Fusion"] == "HardBlend"].copy(), f"{args.ablation_name}_HardBlend")
            soft_summary = summarize_results(df_all[df_all["Fusion"] == "SoftBlend"].copy(), f"{args.ablation_name}_SoftBlend")
            df_final = pd.concat([
                df_all,
                pd.DataFrame(hard_summary),
                pd.DataFrame(soft_summary),
            ], ignore_index=True)
            df_final.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logger.info(f"Quick metrics saved: {csv_path}")
            logger.info(f"Hard images saved : {hard_dir}")
            logger.info(f"Soft images saved : {soft_dir}")
            logger.info(f"Raw images saved  : {raw_dir}")
        else:
            logger.error(f"{dataset_name}: no images evaluated.")


if __name__ == "__main__":
    main()
