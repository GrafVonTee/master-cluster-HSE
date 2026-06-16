#!/usr/bin/env python3
"""Write train/eval matrices for Clingo v3_100 small-model experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_SPECS = {
    "0_6b": {
        "selected_model": "0_6b",
        "model_path": "/workspace/models/qwen3-0.6b",
        "slug": "qwen3-0.6b",
        "config": "configs/train/lora_clingo_synthetic_v3_100_0_6b.yaml",
    },
    "1_7b": {
        "selected_model": "1_7b",
        "model_path": "/workspace/models/qwen3-1.7b",
        "slug": "qwen3-1.7b",
        "config": "configs/train/lora_clingo_synthetic_v3_100_1_7b.yaml",
    },
}

TRAIN_RUNS = ["sft_clingo", "clingo_difficulty_distribution"]
VARIANTS = [
    ("base", None),
    ("sft", "sft_clingo"),
    ("cl", "clingo_difficulty_distribution"),
]


def write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/clingo_v3_100_small")
    parser.add_argument("--models", default="0_6b,1_7b")
    parser.add_argument("--prefix", default="eval")
    args = parser.parse_args()

    root = Path(args.root)
    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in selected if m not in MODEL_SPECS]
    if unknown:
        raise SystemExit(f"Unknown model aliases: {unknown}. Available: {sorted(MODEL_SPECS)}")

    manifest = {"root": str(root), "models": {}, "train_runs": TRAIN_RUNS}
    for alias in selected:
        spec = MODEL_SPECS[alias]
        model_dir = root / alias
        train_file = model_dir / "train_runs.txt"
        eval_file = model_dir / "eval_matrix.tsv"

        write_lines(train_file, TRAIN_RUNS)

        eval_rows: list[str] = []
        for variant, run_name in VARIANTS:
            exp_name = f"{args.prefix}_{alias}_{variant}"
            adapter = ""
            if run_name:
                adapter = f"/workspace/models/{spec['slug']}-sft-{run_name}"
            eval_rows.append("\t".join([exp_name, spec["model_path"], adapter]))
        write_lines(eval_file, eval_rows)

        manifest["models"][alias] = {
            **spec,
            "train_runs_file": str(train_file),
            "eval_matrix_file": str(eval_file),
            "eval_experiments": [r.split("\t", 1)[0] for r in eval_rows],
        }

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
