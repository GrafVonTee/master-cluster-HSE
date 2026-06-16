#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

from datasets import Dataset, load_from_disk


def normalize_row(row: dict) -> dict:
    difficulty = str(row.get("difficulty", "")).strip()
    topic = str(row.get("topic", "")).strip()

    if difficulty not in {"easy", "medium", "hard"}:
        raise ValueError(f"Unexpected difficulty={difficulty!r} for task_id={row.get('task_id')}")

    reference = str(row.get("reference") or row.get("output") or "").strip()
    if not reference:
        raise ValueError(f"Empty reference for task_id={row.get('task_id')}")

    out = dict(row)
    out["output"] = reference
    out["reference"] = reference
    out["difficulty_category"] = difficulty
    out["topic_category"] = topic
    out["length_chars"] = len(reference)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="datasets/clingo/synthetic_v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="datasets/clingo/synthetic_v2_lora_train")
    args = parser.parse_args()

    ds = load_from_disk(args.source)
    if hasattr(ds, "keys"):
        ds = ds[args.split]

    rows = [normalize_row(dict(r)) for r in ds]
    out = Dataset.from_list(rows)
    out.save_to_disk(args.out)

    print(out)
    print("out", args.out)
    print("difficulty", Counter(out["difficulty_category"]))
    print("topic", Counter(out["topic_category"]))
    print("sample", out[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
