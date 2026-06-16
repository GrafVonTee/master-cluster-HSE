#!/usr/bin/env python3
"""Build PythonCodes small-model summary with macro/weighted averages and deltas."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_COLS = ["greedy@1", "pass@1 (n=1)", "mean_%passed", "mean_entropy"]


def load_model_summaries(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(root.glob("*/all_eval/lora_eval_summary.csv")):
        model = csv_path.parent.parent.name
        df = pd.read_csv(csv_path)
        df.insert(0, "model", model)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No merged eval summaries found under {root}/*/all_eval/lora_eval_summary.csv")
    return pd.concat(frames, ignore_index=True)


def add_averages(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in METRIC_COLS if c in df.columns]
    rows = [r.to_dict() for _, r in df.iterrows()]

    for (model, exp), group in df.groupby(["model", "experiment"], dropna=False):
        macro: dict[str, Any] = {
            "model": model,
            "experiment": exp,
            "benchmark": "macro_avg",
            "num_tasks": group["num_tasks"].sum() if "num_tasks" in group.columns else "",
            "model_path": group["model_path"].iloc[0] if "model_path" in group.columns and len(group) else "",
            "adapter_path": group["adapter_path"].iloc[0] if "adapter_path" in group.columns and len(group) else "",
        }
        weighted = dict(macro)
        weighted["benchmark"] = "weighted_avg"

        weights = group["num_tasks"].astype(float) if "num_tasks" in group.columns else None
        for metric in metric_cols:
            macro[metric] = group[metric].astype(float).mean()
            if weights is not None and weights.sum() > 0:
                weighted[metric] = (group[metric].astype(float) * weights).sum() / weights.sum()
            else:
                weighted[metric] = group[metric].astype(float).mean()
        rows.append(macro)
        rows.append(weighted)

    return pd.DataFrame(rows)


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in METRIC_COLS if c in df.columns]
    rows: list[dict[str, Any]] = []
    for _, group in df.groupby(["model", "benchmark"], dropna=False):
        base = group[group["experiment"] == "base"]
        sft = group[group["experiment"] == "sft_pythoncodes"]
        base_row = base.iloc[0] if len(base) else None
        sft_row = sft.iloc[0] if len(sft) else None
        for _, row in group.iterrows():
            item = row.to_dict()
            for metric in metric_cols:
                if base_row is not None and pd.notna(row.get(metric)) and pd.notna(base_row.get(metric)):
                    item[f"{metric}_delta_vs_base"] = float(row[metric]) - float(base_row[metric])
                if sft_row is not None and pd.notna(row.get(metric)) and pd.notna(sft_row.get(metric)):
                    item[f"{metric}_delta_vs_sft"] = float(row[metric]) - float(sft_row[metric])
            rows.append(item)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/pipeline_python/small_models")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_model_summaries(root)
    raw.to_csv(out_dir / "python_small_eval_raw.csv", index=False)

    with_avgs = add_averages(raw)
    comparison = add_deltas(with_avgs)
    comparison = comparison.sort_values(["model", "benchmark", "experiment"], kind="stable")
    comparison.to_csv(out_dir / "python_small_eval_comparison.csv", index=False)
    comparison.to_markdown(out_dir / "python_small_eval_comparison.md", index=False)

    for bench in sorted(comparison["benchmark"].dropna().unique()):
        safe = str(bench).replace("/", "_").replace(" ", "_")
        comparison[comparison["benchmark"] == bench].to_csv(out_dir / f"python_small_{safe}.csv", index=False)

    print(f"wrote {out_dir / 'python_small_eval_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
