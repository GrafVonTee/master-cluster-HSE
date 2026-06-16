#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


def first_existing(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def normalize_tests(row: dict[str, Any]) -> list[str]:
    tests = first_existing(row, ["test_list", "tests", "test", "assertions"])
    if tests is None:
        tests_list = []
    elif isinstance(tests, list):
        tests_list = [str(x).strip() for x in tests if str(x).strip()]
    else:
        tests_list = [line.strip() for line in str(tests).splitlines() if line.strip()]

    setup_parts: list[str] = []

    imports = row.get("test_imports")
    if isinstance(imports, list):
        setup_parts.extend(str(x).strip() for x in imports if str(x).strip())

    setup = row.get("test_setup_code")
    if setup:
        setup_parts.append(str(setup).strip())

    setup_code = "\n".join(setup_parts).strip()

    if setup_code:
        tests_list = [setup_code + "\n" + t for t in tests_list]

    return tests_list


def load_mbpp_any(source: str, config: str | None):
    if config:
        return load_dataset(source, config)
    return load_dataset(source)


def flatten_splits(ds) -> Dataset:
    if isinstance(ds, Dataset):
        return ds
    if isinstance(ds, DatasetDict):
        parts = []
        for split in ds:
            parts.append(ds[split])
        return concatenate_datasets(parts)
    raise TypeError(type(ds))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="datasets/mbpp_grpo_train")
    parser.add_argument("--source", default="google-research-datasets/mbpp")
    parser.add_argument("--config", default="")
    parser.add_argument("--min-task-id", type=int, default=601)
    parser.add_argument("--max-task-id", type=int, default=974)
    args = parser.parse_args()

    fallbacks = [
        (args.source, args.config or None),
        ("google-research-datasets/mbpp", None),
        ("Muennighoff/mbpp", None),
        ("RLAIF/mbpp", None),
        ("claudios/google-research-datasets__mbpp", None),
    ]

    last_error = None
    raw = None
    used = None

    for source, config in fallbacks:
        try:
            raw = flatten_splits(load_mbpp_any(source, config))
            used = (source, config)
            break
        except Exception as e:
            last_error = e

    if raw is None:
        raise RuntimeError(f"Could not load MBPP. Last error: {last_error!r}")

    rows = []
    for row in raw:
        task_id_raw = first_existing(row, ["task_id", "id"])
        try:
            task_id = int(task_id_raw)
        except Exception:
            continue

        # Google MBPP train split by task_id: 601-974.
        if not (args.min_task_id <= task_id <= args.max_task_id):
            continue

        instruction = first_existing(row, ["prompt", "text", "instruction", "question"])
        code = first_existing(row, ["code", "output", "solution", "canonical_solution"])
        tests = normalize_tests(row)

        if not instruction or not code or not tests:
            continue

        rows.append({
            "task_id": str(task_id),
            "instruction": str(instruction).strip(),
            "input": "",
            "output": str(code).strip(),
            "reference": str(code).strip(),
            "tests": tests,
            "source": used[0],
        })

    if not rows:
        raise RuntimeError("Prepared MBPP rows = 0")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(out))

    print("source", used)
    print("rows", len(ds))
    print("columns", ds.column_names)
    print("out", out)
    print("first", ds[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
