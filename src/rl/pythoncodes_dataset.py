from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from src.data.prompt.pythoncodes_cl_scored import build_messages, build_prompt
from src.rl.code import normalize_tests


def _path_or_none(value: Any) -> Path | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return Path(s)


def load_pythoncodes_source(dataset_cfg: dict[str, Any]) -> Dataset:
    split = str(dataset_cfg.get("split", "train"))
    parquet_path = _path_or_none(dataset_cfg.get("parquet_path"))
    disk_path = _path_or_none(dataset_cfg.get("disk_path"))

    if parquet_path and parquet_path.exists():
        ds = load_dataset("parquet", data_files=str(parquet_path), split="train")
    elif disk_path and disk_path.exists():
        loaded = load_from_disk(str(disk_path))
        if isinstance(loaded, DatasetDict):
            ds = loaded[split] if split in loaded else loaded["train"]
        else:
            ds = loaded
    else:
        raise FileNotFoundError(
            f"Could not find pythoncodes dataset. disk_path={disk_path}, parquet_path={parquet_path}"
        )

    limit = dataset_cfg.get("limit")
    if limit is not None:
        limit = int(limit)
        if limit > 0:
            ds = ds.select(range(min(limit, len(ds))))
    seed = int(dataset_cfg.get("seed", 42))
    ds = ds.shuffle(seed=seed)
    return ds


def _first_existing(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def prepare_pythoncodes_grpo_dataset(dataset_cfg: dict[str, Any], tokenizer=None) -> Dataset:
    ds = load_pythoncodes_source(dataset_cfg)
    prompt_format = str(dataset_cfg.get("prompt_format", "chat")).strip().lower()
    min_reference_chars = int(dataset_cfg.get("min_reference_chars", 0) or 0)

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        reference = str(_first_existing(row, ["output", "canonical_solution", "solution", "code"]) or "").strip()
        tests = _first_existing(row, ["tests", "test", "assertions", "unit_tests", "test_list"])
        tests_list = normalize_tests(tests)

        if prompt_format == "text":
            if tokenizer is None:
                # Fall back to a plain instruction prompt if no tokenizer was provided.
                prompt = str(row.get("instruction") or row.get("text") or "").strip()
                inp = str(row.get("input") or "").strip()
                if inp:
                    prompt = f"{prompt}\n\nInput:\n{inp}" if prompt else f"Input:\n{inp}"
            else:
                prompt = build_prompt(row, tokenizer=tokenizer, train=False)["text"]
        else:
            prompt = build_messages(row, train=False)["messages"]

        return {
            "prompt": prompt,
            "reference": reference,
            "tests": tests_list,
            "task_id": str(row.get("task_id") or row.get("id") or row.get("problem_id") or ""),
        }

    keep = ["prompt", "reference", "tests", "task_id"]
    ds = ds.map(convert, remove_columns=list(ds.column_names), desc="Preparing pythoncodes GRPO dataset")
    if min_reference_chars > 0:
        ds = ds.filter(lambda row: len(str(row.get("reference") or "")) >= min_reference_chars)
    return ds.select_columns(keep)
