#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="grpo_pythoncodes")
    parser.add_argument("--benchmarks", default="mbpp,humaneval")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--include-base", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "scripts/eval_lora_all.py",
        "--benchmarks", args.benchmarks,
        "--limit", str(args.limit),
        "--only", args.run_name,
    ]
    if args.include_base:
        cmd.insert(-2, "--include-base")
    print("[rl-eval]", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
