# -*- coding: utf-8 -*-
"""
Create test yml files for module ablation checkpoints.

Use after strict module variants have been trained.
"""

import argparse
import json
import re
from pathlib import Path


def patch_yml(text, name, weight, prompt_path):
    s = text
    if re.search(r"(?m)^name:\s*.*$", s):
        s = re.sub(r"(?m)^name:\s*.*$", f"name: {name}", s)
    else:
        s = f"name: {name}\n" + s
    s = re.sub(r"(?m)^(\s*)pretrain_model_G:\s*.*$", rf"\1pretrain_model_G: {weight}", s)
    if prompt_path:
        if re.search(r"(?m)^(\s*)prompt_path:\s*.*$", s):
            s = re.sub(r"(?m)^(\s*)prompt_path:\s*.*$", rf"\1prompt_path: {prompt_path}", s)
        else:
            s += f"\nprompt_path: {prompt_path}\n"
    return s


def parse_variant(s):
    # format: name=/path/to/weight.pth
    if "=" not in s:
        raise ValueError(f"Variant must be name=weight_path, got: {s}")
    name, path = s.split("=", 1)
    return name.strip(), path.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-yml", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompt-path", required=True, help="Flat full_structured prompt json.")
    parser.add_argument("--variant", action="append", default=[], help="name=/path/to/weight.pth. Can be repeated.")
    args = parser.parse_args()

    base = Path(args.base_yml).read_text(encoding="utf-8")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    for v in args.variant:
        name, weight = parse_variant(v)
        yml = out_dir / f"test_module_{name}.yml"
        yml.write_text(patch_yml(base, f"module_ablation_{name}", weight, args.prompt_path), encoding="utf-8")
        commands.append(f"python test_combined_final.py -opt {yml.resolve()}")

    (out_dir / "test_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote ymls to {out_dir}")
    print((out_dir / "test_commands.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
