#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric", default="pass@1 (n=1)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.metric not in df.columns:
        raise SystemExit(f"Missing metric column: {args.metric}")

    rows = []
    for exp, g in df.groupby("experiment"):
        if not str(exp).startswith("cl_"):
            continue
        macro = float(g[args.metric].mean())
        weighted = float((g[args.metric] * g["num_tasks"]).sum() / g["num_tasks"].sum())
        rows.append({
            "experiment": exp,
            "macro_pass1": macro,
            "weighted_pass1": weighted,
            "benchmarks": ",".join(sorted(map(str, g["benchmark"].unique()))),
        })

    if not rows:
        raise SystemExit("No CL experiments found")

    rank = pd.DataFrame(rows).sort_values(["macro_pass1", "weighted_pass1"], ascending=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rank.to_csv(out.parent / "cl_rank.csv", index=False)
    rank.to_markdown(out.parent / "cl_rank.md", index=False)

    best = str(rank.iloc[0]["experiment"])
    out.write_text(best + "\n", encoding="utf-8")
    print("best_cl", best)
    print(rank.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
