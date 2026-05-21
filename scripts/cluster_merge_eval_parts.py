#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import pandas as pd


METRIC_COLS = ["greedy@1", "pass@1 (n=1)", "mean_%passed", "mean_entropy"]


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
    summary = summary.drop_duplicates(subset=["experiment", "benchmark"], keep="last")
    summary = summary.sort_values(["benchmark", "experiment"], kind="stable")

    summary_csv = out_dir / "lora_eval_summary.csv"
    summary_md = out_dir / "lora_eval_summary.md"
    summary_json = out_dir / "lora_eval_summary.json"
    summary.to_csv(summary_csv, index=False)
    summary.to_markdown(summary_md, index=False)
    summary.to_json(summary_json, orient="records", indent=2, force_ascii=False)

    metric_cols = [c for c in METRIC_COLS if c in summary.columns]
    comparison_rows = []
    for benchmark, group in summary.groupby("benchmark", dropna=False):
        base = group[group["experiment"] == "base"]
        sft = group[group["experiment"] == "sft_pythoncodes"]
        base_vals = base.iloc[0].to_dict() if len(base) else {}
        sft_vals = sft.iloc[0].to_dict() if len(sft) else {}
        for _, row in group.iterrows():
            item = row.to_dict()
            for col in metric_cols:
                if col in base_vals and pd.notna(row[col]) and pd.notna(base_vals[col]):
                    item[f"{col}_delta_vs_base"] = float(row[col]) - float(base_vals[col])
                if col in sft_vals and pd.notna(row[col]) and pd.notna(sft_vals[col]):
                    item[f"{col}_delta_vs_sft"] = float(row[col]) - float(sft_vals[col])
            comparison_rows.append(item)

    comp = pd.DataFrame(comparison_rows)
    comp_csv = out_dir / "lora_eval_comparison.csv"
    comp_md = out_dir / "lora_eval_comparison.md"
    comp.to_csv(comp_csv, index=False)
    comp.to_markdown(comp_md, index=False)

    for metric in metric_cols:
        pivot = summary.pivot_table(index="experiment", columns="benchmark", values=metric, aggfunc="first").reset_index()
        safe_metric = metric.replace("@", "at").replace("%", "pct").replace(" ", "_").replace("(", "").replace(")", "")
        pivot.to_csv(out_dir / f"pivot_{safe_metric}.csv", index=False)
        pivot.to_markdown(out_dir / f"pivot_{safe_metric}.md", index=False)

    report = {
        "parts_found": len(files),
        "rows": len(summary),
        "summary_csv": str(summary_csv),
        "comparison_csv": str(comp_csv),
    }
    (out_dir / "eval_merge_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
