#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk


def md5(x: str) -> str:
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="datasets/clingo/synthetic_v2")
    parser.add_argument("--out", default="datasets/clingo/synthetic_v2_grouped")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-groups-per-topic", type=int, default=2)
    parser.add_argument("--val-groups-per-topic", type=int, default=2)
    args = parser.parse_args()

    ds = load_from_disk(args.source)
    if isinstance(ds, DatasetDict):
        all_ds = concatenate_datasets([ds[k] for k in ds.keys()])
    else:
        all_ds = ds

    rows = [dict(r) for r in all_ds]
    print("total rows", len(rows))

    # Deduplicate exact rows just in case.
    seen = set()
    dedup = []
    for r in rows:
        key = (
            r.get("task_id"),
            r.get("topic"),
            r.get("instruction"),
            r.get("facts"),
            r.get("reference") or r.get("output"),
        )
        if key not in seen:
            seen.add(key)
            dedup.append(r)
    rows = dedup
    print("dedup rows", len(rows))

    by_topic_group = defaultdict(lambda: defaultdict(list))
    for r in rows:
        topic = str(r["topic"])
        # Critical group key: same topic + same facts must stay in one split.
        g = md5(topic + "\n" + str(r.get("facts", "")))
        by_topic_group[topic][g].append(r)

    rng = random.Random(args.seed)
    train, val, test = [], [], []

    print("\nGroup counts by topic:")
    for topic in sorted(by_topic_group):
        groups = list(by_topic_group[topic].items())
        rng.shuffle(groups)

        n_groups = len(groups)
        n_test = min(args.test_groups_per_topic, max(1, n_groups // 5))
        n_val = min(args.val_groups_per_topic, max(1, n_groups // 5))

        test_groups = groups[:n_test]
        val_groups = groups[n_test:n_test + n_val]
        train_groups = groups[n_test + n_val:]

        print(
            topic,
            "groups", n_groups,
            "train_groups", len(train_groups),
            "val_groups", len(val_groups),
            "test_groups", len(test_groups),
            "rows/train/val/test",
            sum(len(x) for _, x in train_groups),
            sum(len(x) for _, x in val_groups),
            sum(len(x) for _, x in test_groups),
        )

        for _, bucket in train_groups:
            train.extend(bucket)
        for _, bucket in val_groups:
            val.extend(bucket)
        for _, bucket in test_groups:
            test.extend(bucket)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    out = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test),
    })
    out.save_to_disk(args.out)

    print("\nSaved", args.out)
    for split in ["train", "validation", "test"]:
        d = out[split]
        print("\n", split, len(d))
        print("topic", Counter(d["topic"]))
        print("difficulty", Counter(d["difficulty"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
