#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except Exception as e:
    raise SystemExit(
        "qwen_vl_utils is required in the labeling environment. "
        f"Import error: {e}"
    )

SYSTEM = (
    'Describe only visible evidence in the supplied masked Dunhuang mural image '
    'and binary mask. Return one concise English description using the fields '
    '"Visible content", "Painted style and palette", and "Mask morphology". '
    'Do not infer or name content hidden by the mask.'
)

USER_TEXT = (
    "The first image is the masked mural image. The second image is the binary mask "
    "(white = known, black = missing). Produce the description in the required "
    "three-field format."
)


def locate_model(root: Path) -> Path:
    snaps = sorted((root / "snapshots").glob("*"))
    candidates = [p for p in snaps if (p / "config.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No Qwen2-VL snapshot under {root}/snapshots")
    return candidates[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="One damage class root containing test_masked/test_mask.")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--model-root",
        default="/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct",
    )
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_root = Path(args.data_root)
    masked_root = data_root / "test_masked"
    mask_root = data_root / "test_mask"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    model_path = locate_model(Path(args.model_root))
    print("Qwen model:", model_path)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )

    records = {}

    for image_path in sorted(masked_root.glob('*_masked.png')):
        idx = image_path.stem.replace('_masked', '')
        mask_path = mask_root / f"{idx}_mask.png"
        assert image_path.exists(), image_path
        assert mask_path.exists(), mask_path

        messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "image", "image": str(mask_path)},
                    {"type": "text", "text": USER_TEXT},
                ],
            },
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated)
        ]
        prompt = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        records[str(idx)] = {
            "image_id": str(idx),
            "masked_image": str(image_path),
            "mask": str(mask_path),
            "final_prompt": prompt,
        }

        print(f"[{idx}] {prompt}")

        # Crash-safe checkpoint after every item.
        out.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("Saved:", out)
    print("N:", len(records))


if __name__ == "__main__":
    main()
