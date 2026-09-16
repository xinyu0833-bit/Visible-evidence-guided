# -*- coding: utf-8 -*-
"""Run yml-based ablation tests with the existing test_combined_final.py."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-script", required=True)
    parser.add_argument("--yml-dir", required=True)
    parser.add_argument("--only", default="", help="Comma-separated substrings. Only yml files whose names contain one of them are run.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default="python")
    args = parser.parse_args()

    yml_dir = Path(args.yml_dir)
    ymls = sorted((yml_dir / "ymls").glob("*.yml")) if (yml_dir / "ymls").exists() else sorted(yml_dir.glob("*.yml"))

    if args.only.strip():
        keys = [x.strip() for x in args.only.split(",") if x.strip()]
        ymls = [p for p in ymls if any(k in p.name for k in keys)]

    if not ymls:
        raise SystemExit(f"No yml files found in {yml_dir}")

    for yml in ymls:
        cmd = [args.python, args.test_script, "-opt", str(yml)]
        print("[CMD]", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
