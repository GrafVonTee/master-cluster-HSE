#!/usr/bin/env python3
"""Host-side local launcher for training/eval Docker services."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check)


def dc() -> list[str]:
    return ["docker", "compose"]


def pick(cli_value, env_name: str, default=None):
    if cli_value is not None:
        return cli_value
    value = os.environ.get(env_name)
    if value is None or str(value).strip() == "":
        return default
    return value


def env_args(mapping: dict[str, object | None]) -> list[str]:
    out: list[str] = []
    for key, value in mapping.items():
        if value is None or str(value).strip() == "":
            continue
        out.extend(["-e", f"{key}={value}"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/lora_pythoncodes_cl.yaml")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--eval-benchmarks", default="mbpp,humaneval")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-max-new-tokens", type=int, default=None)
    parser.add_argument("--train-max-steps-override", type=int, default=None)
    parser.add_argument("--no-base", action="store_true")
    args = parser.parse_args()

    compose = dc()
    train_steps = pick(args.train_max_steps_override, "TRAIN_MAX_STEPS_OVERRIDE")
    eval_new_tokens = pick(args.eval_max_new_tokens, "EVAL_MAX_NEW_TOKENS", 512)

    if args.rebuild:
        run(compose + ["build", "train", "eval", "tensorboard"])

    if not args.no_tensorboard:
        run(compose + ["up", "-d", "tensorboard"])
        print("TensorBoard: http://localhost:6006", flush=True)

    if not args.skip_train:
        cmd = ["python", "scripts/train_cl_lora_all.py", "--config", args.config]
        if args.force:
            cmd.append("--force")
        if args.only:
            cmd += ["--only", *args.only]
        run(compose + ["run", "--rm", *env_args({"TRAIN_MAX_STEPS_OVERRIDE": train_steps}), "train"] + cmd)

    if not args.skip_eval:
        cmd = ["python", "scripts/eval_lora_all.py", "--benchmarks", args.eval_benchmarks]
        if not args.no_base:
            cmd.append("--include-base")
        if args.eval_limit is not None:
            cmd += ["--limit", str(args.eval_limit)]
        if args.only:
            cmd += ["--only", *args.only]
        run(compose + ["run", "--rm", *env_args({"EVAL_MAX_NEW_TOKENS": eval_new_tokens}), "eval"] + cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
