# -*- coding: utf-8 -*-
"""
Create training yml files for strict module ablations from an existing full training yml.

Required strict module ablations:
1. w/o Boundary Loss: lambda_boundary = 0
2. w/o Routing: route_mode = global

Optional:
3. hole_only_routing
4. no_prompt_adapter

The script uses PyYAML if available. It preserves the semantics of your original yml
but may not preserve comments.
"""

import argparse
import copy
from pathlib import Path

try:
    import yaml
except Exception as e:
    yaml = None


def set_nested(d, keys, value):
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def recursively_set_key(obj, target_key, value):
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k == target_key:
                obj[k] = value
            else:
                recursively_set_key(obj[k], target_key, value)
    elif isinstance(obj, list):
        for x in obj:
            recursively_set_key(x, target_key, value)


def update_common(cfg, name, train_prompt_path=None, pretrain_backbone=None):
    cfg["name"] = name
    if pretrain_backbone:
        set_nested(cfg, ["path", "pretrain_model_G"], pretrain_backbone)
    if train_prompt_path:
        recursively_set_key(cfg, "prompt_path", train_prompt_path)
    return cfg


def make_variant(base, name, route_mode=None, lambda_boundary=None, use_prompt_adapter=None, prompt_injection=None, adapter_variant=None):
    cfg = copy.deepcopy(base)
    cfg["name"] = name

    if route_mode is not None:
        set_nested(cfg, ["network_G", "setting", "route_mode"], route_mode)
    if use_prompt_adapter is not None:
        set_nested(cfg, ["network_G", "setting", "use_prompt_adapter"], bool(use_prompt_adapter))
    if prompt_injection is not None:
        set_nested(cfg, ["network_G", "setting", "prompt_injection"], prompt_injection)
    if adapter_variant is not None:
        set_nested(cfg, ["network_G", "setting", "adapter_variant"], adapter_variant)

    if lambda_boundary is not None:
        set_nested(cfg, ["train", "lambda_boundary"], float(lambda_boundary))
    return cfg


def dump_yaml(cfg, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-train-yml", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exp-prefix", default="StrDiffusion_Dunhuang_ModuleAblation")
    parser.add_argument("--train-prompt-path", default="")
    parser.add_argument("--pretrain-backbone", default="")
    parser.add_argument("--mode-set", choices=["minimal", "full"], default="minimal")
    parser.add_argument("--train-script", default="train.py")
    args = parser.parse_args()

    if yaml is None:
        raise SystemExit("PyYAML is required. Install with: pip install pyyaml")

    base_yml = Path(args.base_train_yml)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(base_yml, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    base = update_common(
        base,
        name=args.exp_prefix + "_base",
        train_prompt_path=args.train_prompt_path or None,
        pretrain_backbone=args.pretrain_backbone or None,
    )

    variants = {
        "train_full_reference.yml": make_variant(
            base,
            f"{args.exp_prefix}_FullReference",
            route_mode="hole_boundary",
            lambda_boundary=2.0,
            use_prompt_adapter=True,
            prompt_injection="full",
            adapter_variant="film_attn",
        ),
        "train_no_boundary.yml": make_variant(
            base,
            f"{args.exp_prefix}_NoBoundaryLoss",
            route_mode="hole_boundary",
            lambda_boundary=0.0,
            use_prompt_adapter=True,
            prompt_injection="full",
            adapter_variant="film_attn",
        ),
        "train_no_routing_global.yml": make_variant(
            base,
            f"{args.exp_prefix}_NoRoutingGlobalInjection",
            route_mode="global",
            lambda_boundary=2.0,
            use_prompt_adapter=True,
            prompt_injection="full",
            adapter_variant="film_attn",
        ),
    }

    if args.mode_set == "full":
        variants.update({
            "train_hole_only_routing.yml": make_variant(
                base,
                f"{args.exp_prefix}_HoleOnlyRouting",
                route_mode="hole_only",
                lambda_boundary=2.0,
                use_prompt_adapter=True,
                prompt_injection="full",
                adapter_variant="film_attn",
            ),
            "train_no_prompt_adapter.yml": make_variant(
                base,
                f"{args.exp_prefix}_NoPromptAdapter",
                route_mode="global",
                lambda_boundary=2.0,
                use_prompt_adapter=False,
                prompt_injection="full",
                adapter_variant="none",
            ),
        })

    commands = []
    for fname, cfg in variants.items():
        path = out_dir / fname
        dump_yaml(cfg, path)
        commands.append(f"python {args.train_script} -opt {path.resolve()}")

    cmd_file = out_dir / "train_commands.txt"
    cmd_file.write_text("\n".join(commands) + "\n", encoding="utf-8")

    print(f"[INFO] Wrote {len(variants)} train yml files to {out_dir}")
    print(f"[INFO] Wrote commands to {cmd_file}")
    print(cmd_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
