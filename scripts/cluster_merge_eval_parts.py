#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-root", default="outputs/eval_jobs")
    parser.add_argument("--out-dir", default="outputs/eval")
    args = parser.parse_args()

    parts_root = Path(args.parts_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(parts_root.glob("*/eval/lora_eval_summary.csv"))
    if not files:
        raise SystemExit(f"No eval summary parts found under {parts_root}/*/eval/lora_eval_summary.csv")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["part_dir"] = path.parent.parent.name
        frames.append(df)

    summary = pd.concat(frames, ignore_index=True)
    summary = summary.sort_values(["benchmark", "experiment"], kind="stable")

    summary_csv = out_dir / "lora_eval_summary.csv"
    summary_md = out_dir / "lora_eval_summary.md"
    summary.to_csv(summary_csv, index=False)
    summary.to_markdown(summary_md, index=False)

    metric_cols = [c for c in ["greedy@1", "pass@1 (n=1)", "mean_%passed", "mean_entropy"] if c in summary.columns]
    comparison_rows = []
    for benchmark, group in summary.groupby("benchmark", dropna=False):
        base = group[group["experiment"] == "base"]
        sft = group[group["experiment"] == "sft_pythoncodes"]
        base_vals = base.iloc[0].to_dict() if len(base) else {}
        sft_vals = sft.iloc[0].to_dict() if len(sft) else {}
        for _, row in group.iterrows():
            item = row.to_dict()
            for col in metric_cols:
                if col in base_vals:
                    item[f"delta_vs_base__{col}"] = row[col] - base_vals[col]
                if col in sft_vals:
                    item[f"delta_vs_sft__{col}"] = row[col] - sft_vals[col]
            comparison_rows.append(item)

    comp = pd.DataFrame(comparison_rows)
    comp_csv = out_dir / "lora_eval_comparison.csv"
    comp_md = out_dir / "lora_eval_comparison.md"
    comp.to_csv(comp_csv, index=False)
    comp.to_markdown(comp_md, index=False)

    print(f"Merged {len(files)} parts")
    print(f"Saved {summary_csv}")
    print(f"Saved {summary_md}")
    print(f"Saved {comp_csv}")
    print(f"Saved {comp_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
