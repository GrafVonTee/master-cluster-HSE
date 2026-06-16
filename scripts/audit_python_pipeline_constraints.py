#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def get_sbatch(text: str, key: str) -> str | None:
    pat = rf"^#SBATCH\s+{re.escape(key)}(?:=|\s+)(.+)$"
    m = re.search(pat, text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def check_job(path: Path, *, gpus: str | None, cpus: str, partition: str = "rocky") -> list[str]:
    errs = []
    text = read(path)
    part = get_sbatch(text, "--partition")
    if part != partition:
        errs.append(f"{path}: partition={part!r}, expected {partition!r}")
    c = get_sbatch(text, "--cpus-per-task")
    if c != cpus:
        errs.append(f"{path}: cpus={c!r}, expected {cpus!r}")
    if gpus is not None:
        gg = get_sbatch(text, "--gpus")
        gres = get_sbatch(text, "--gres")
        if gg != gpus:
            errs.append(f"{path}: gpus={gg!r}, expected {gpus!r}")
        if gres is not None:
            errs.append(f"{path}: has --gres={gres!r}; expected only --gpus")
    return errs


def main() -> int:
    root = Path.cwd()
    errs: list[str] = []

    for path in [root / "jobs/python_train_matrix_rocky.sbatch", root / "jobs/python_eval_matrix_rocky.sbatch"]:
        if not path.exists():
            errs.append(f"missing {path}")
        else:
            errs += check_job(path, gpus="1", cpus="2")

    merge = root / "jobs/python_merge_eval_rocky.sbatch"
    if merge.exists():
        errs += check_job(merge, gpus=None, cpus="2")
    else:
        errs.append(f"missing {merge}")

    grpo_candidates = sorted((root / "jobs").glob("rl_v100_vllm085_grpo_server*2gpu*.sbatch"))
    if not grpo_candidates:
        errs.append("missing 2GPU GRPO job: jobs/rl_v100_vllm085_grpo_server*2gpu*.sbatch")
    for path in grpo_candidates:
        errs += check_job(path, gpus="2", cpus="4")
        text = read(path)
        if 'RL_MAX_STEPS_OVERRIDE="${RL_MAX_STEPS_OVERRIDE:-1}"' in text:
            errs.append(f"{path}: RL_MAX_STEPS_OVERRIDE still defaults to 1")
        if 'RL_DATASET_LIMIT_OVERRIDE="${RL_DATASET_LIMIT_OVERRIDE:-64}"' in text:
            errs.append(f"{path}: RL_DATASET_LIMIT_OVERRIDE still defaults to 64")
        if "SINGULARITYENV_GRPO_ADAPTER_OUTPUT_DIR" not in text:
            errs.append(f"{path}: does not propagate GRPO_ADAPTER_OUTPUT_DIR")

    train_cfg = root / "configs/train/lora_pythoncodes_cl.yaml"
    if train_cfg.exists():
        text = read(train_cfg)
        if not re.search(r"(?m)^\s*dataloader_num_workers:\s*2\s*$", text):
            errs.append(f"{train_cfg}: dataloader_num_workers must be 2")
    else:
        errs.append(f"missing {train_cfg}")

    grpo_cfg = root / "configs/rl/grpo_pythoncodes.yaml"
    if grpo_cfg.exists():
        text = read(grpo_cfg)
        for key, val in [("limit", "1024"), ("max_steps", "100"), ("max_completion_length", "256"), ("dataloader_num_workers", "0")]:
            if not re.search(rf"(?m)^\s*{key}:\s*{val}\s*$", text):
                errs.append(f"{grpo_cfg}: expected {key}: {val}")
    else:
        errs.append(f"missing {grpo_cfg}")

    grpo_py = root / "scripts/rl_train_pythoncodes_grpo_server.py"
    if grpo_py.exists():
        text = read(grpo_py)
        if "base_slug = Path(str(base_model)).name.rstrip" not in text:
            errs.append(f"{grpo_py}: adapter output dir is not dynamic by base model slug")
    else:
        errs.append(f"missing {grpo_py}")

    if errs:
        print("AUDIT FAIL")
        for e in errs:
            print(" -", e)
        return 1

    print("AUDIT OK")
    print("SFT/CL jobs: 1 GPU, 2 CPU")
    print("Eval jobs:   1 GPU, 2 CPU")
    print("RL jobs:     2 GPU, 4 CPU")
    print("Matrix throttles: train/eval %4; 4B train %2; RL at most two jobs in parallel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
