from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from datasets import Dataset, DatasetDict, load_from_disk

from src.data.prompt.clingo_synthetic import build_messages, build_prompt


def _path_or_none(value: Any) -> Optional[Path]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return Path(s)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    s = str(value).strip()
    return [s] if s else []


def load_clingo_source(dataset_cfg: Dict[str, Any]) -> Dataset:
    split = str(dataset_cfg.get("split", "train"))
    disk_path = _path_or_none(dataset_cfg.get("disk_path"))
    if not disk_path or not disk_path.exists():
        raise FileNotFoundError(f"Could not find clingo dataset. disk_path={disk_path}")

    loaded = load_from_disk(str(disk_path))
    if isinstance(loaded, DatasetDict):
        ds = loaded[split] if split in loaded else loaded["train"]
    else:
        ds = loaded

    topics = _as_list(dataset_cfg.get("topic_filter"))
    if topics:
        wanted = set(topics)
        ds = ds.filter(lambda row: str(row.get("topic", "")) in wanted)

    difficulties = _as_list(dataset_cfg.get("difficulty_filter"))
    if difficulties:
        wanted = set(difficulties)
        ds = ds.filter(lambda row: str(row.get("difficulty", "")) in wanted)

    seed = int(dataset_cfg.get("seed", 42))
    ds = ds.shuffle(seed=seed)

    limit = dataset_cfg.get("limit")
    if limit is not None:
        limit = int(limit)
        if limit > 0:
            ds = ds.select(range(min(limit, len(ds))))
    return ds


def _serializable_task(row: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "task_id",
        "source_task_id",
        "language",
        "topic",
        "difficulty",
        "instruction",
        "facts",
        "output",
        "reference",
        "expected_satisfiable",
        "expected_atoms",
        "forbidden_atoms",
        "oracle_tests",
    ]
    return {k: row.get(k) for k in keep if k in row}


def prepare_clingo_grpo_dataset(dataset_cfg: Dict[str, Any], tokenizer=None) -> Dataset:
    ds = load_clingo_source(dataset_cfg)
    prompt_format = str(dataset_cfg.get("prompt_format", "chat")).strip().lower()

    def convert(row: Dict[str, Any]) -> Dict[str, Any]:
        if prompt_format == "text" and tokenizer is not None:
            prompt = build_prompt(row, tokenizer=tokenizer, train=False)["text"]
        else:
            prompt = build_messages(row, train=False)["messages"]
        task = _serializable_task(dict(row))
        return {
            "prompt": prompt,
            "task": task,
            "task_id": str(row.get("task_id") or row.get("source_task_id") or ""),
            "reference": str(row.get("reference") or row.get("output") or ""),
        }

    keep = ["prompt", "task", "task_id", "reference"]
    ds = ds.map(convert, remove_columns=list(ds.column_names), desc="Preparing clingo GRPO dataset")
    return ds.select_columns(keep)
