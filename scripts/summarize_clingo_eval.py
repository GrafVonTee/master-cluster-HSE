#!/usr/bin/env python3
"""Collect Clingo eval summaries and compute deltas against base/SFT."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_COLS = ["mean_reward", "full_pass_rate", "partial_rate", "error_rate"]


def infer_model_variant(exp: str) -> tuple[str, str]:
    # Expected examples: eval_0_6b_base, eval_1_7b_sft, eval_0_6b_cl.
    m = re.match(r"(?:eval_)?(?P<model>.+)_(?P<variant>base|sft|cl)$", exp)
    if m:
        return m.group("model"), m.group("variant")
    return "unknown", exp


def load_parts(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for summary_path in sorted(root.glob("*/summary.csv")):
        exp = summary_path.parent.name
        df = pd.read_csv(summary_path)
        model, variant = infer_model_variant(exp)
        df.insert(0, "experiment", exp)
        df.insert(1, "model", model)
        df.insert(2, "variant", variant)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No summaries found under {root}/*/summary.csv")
    return pd.concat(frames, ignore_index=True)


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    keys = ["model", "group", "name"]
    for _, group in df.groupby(keys, dropna=False):
        base = group[group["variant"] == "base"]
        sft = group[group["variant"] == "sft"]
        base_row = base.iloc[0] if len(base) else None
        sft_row = sft.iloc[0] if len(sft) else None
        for _, row in group.iterrows():
            item = row.to_dict()
            for metric in [c for c in METRIC_COLS if c in group.columns]:
                if base_row is not None and pd.notna(row.get(metric)) and pd.notna(base_row.get(metric)):
                    item[f"{metric}_delta_vs_base"] = float(row[metric]) - float(base_row[metric])
                if sft_row is not None and pd.notna(row.get(metric)) and pd.notna(sft_row.get(metric)):
                    item[f"{metric}_delta_vs_sft"] = float(row[metric]) - float(sft_row[metric])
            out_rows.append(item)
    return pd.DataFrame(out_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/clingo_v3_100_small")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_parts(root)
    summary = summary.sort_values(["model", "group", "name", "variant", "experiment"], kind="stable")
    summary.to_csv(out_dir / "clingo_eval_summary_raw.csv", index=False)

    comparison = add_deltas(summary)
    comparison = comparison.sort_values(["model", "group", "name", "variant", "experiment"], kind="stable")
    comparison.to_csv(out_dir / "clingo_eval_comparison.csv", index=False)
    comparison.to_markdown(out_dir / "clingo_eval_comparison.md", index=False)

    overall = comparison[comparison["group"] == "overall"].copy()
    overall.to_csv(out_dir / "clingo_eval_overall.csv", index=False)

    by_difficulty = comparison[comparison["group"] == "difficulty"].copy()
    by_difficulty.to_csv(out_dir / "clingo_eval_by_difficulty.csv", index=False)

    by_topic = comparison[comparison["group"] == "topic"].copy()
    by_topic.to_csv(out_dir / "clingo_eval_by_topic.csv", index=False)

    print(f"wrote {out_dir / 'clingo_eval_comparison.csv'}")
    print(f"wrote {out_dir / 'clingo_eval_overall.csv'}")
    print(f"wrote {out_dir / 'clingo_eval_by_difficulty.csv'}")
    print(f"wrote {out_dir / 'clingo_eval_by_topic.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
