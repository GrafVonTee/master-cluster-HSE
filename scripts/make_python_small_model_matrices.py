#!/usr/bin/env python3
"""Write train/eval matrices for PythonCodes 0.6B/1.7B SFT+CL experiments."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


CRITERIA = ["length", "perplexity", "lexical", "semantic", "llm_judge"]
SCHEDULES = ["staged", "cumulative", "distribution"]
TRAIN_RUNS = ["sft_pythoncodes"] + [f"cl_{criterion}_{schedule}" for schedule in SCHEDULES for criterion in CRITERIA]
EVAL_EXPERIMENTS = ["base"] + TRAIN_RUNS

MODEL_SPECS = {
    "0_6b": {
        "selected_model": "0_6b",
        "config": "configs/train/lora_pythoncodes_cl_0_6b.yaml",
    },
    "1_7b": {
        "selected_model": "1_7b",
        "config": "configs/train/lora_pythoncodes_cl_1_7b.yaml",
    },
}


def write_lines(path: Path, rows: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/pipeline_python/small_models")
    parser.add_argument("--models", default="0_6b,1_7b")
    args = parser.parse_args()

    root = Path(args.root)
    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in selected if m not in MODEL_SPECS]
    if unknown:
        raise SystemExit(f"Unknown model aliases: {unknown}. Available: {sorted(MODEL_SPECS)}")

    manifest = {
        "root": str(root),
        "criteria": CRITERIA,
        "schedules": SCHEDULES,
        "train_runs": TRAIN_RUNS,
        "eval_experiments": EVAL_EXPERIMENTS,
        "models": {},
    }

    for alias in selected:
        model_dir = root / alias
        train_file = model_dir / "train_runs.txt"
        eval_file = model_dir / "eval_experiments.txt"
        write_lines(train_file, TRAIN_RUNS)
        write_lines(eval_file, EVAL_EXPERIMENTS)
        manifest["models"][alias] = {
            **MODEL_SPECS[alias],
            "train_runs_file": str(train_file),
            "eval_experiments_file": str(eval_file),
            "parts_root": str(model_dir / "eval_jobs"),
            "merged_eval_dir": str(model_dir / "all_eval"),
        }

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
