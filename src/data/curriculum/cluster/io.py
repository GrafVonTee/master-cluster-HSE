import json
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_DIR = Path(
    os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[4])
).resolve()

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def read_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_DIR / path

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_offline(cfg: dict) -> None:
    if bool(cfg.get("runtime", {}).get("offline", True)):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"


def hf_home() -> Path:
    path = Path(os.environ.get("HF_HOME", PROJECT_DIR / ".cache" / "huggingface"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_cache_dir() -> Path:
    path = PROJECT_DIR / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def out_root(cfg: dict) -> Path:
    path = PROJECT_DIR / cfg["output"]["root"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_path(value: str | None) -> str | None:
    if value is None:
        return None

    path = Path(str(value))
    if path.is_absolute():
        return str(path)

    return str(PROJECT_DIR / path)


def model_path_from_cfg(cfg: dict) -> str:
    local_path = cfg["model"].get("local_path")
    if local_path:
        return resolve_project_path(local_path)

    return cfg["model"]["name_or_path"]


def dtype_from_cfg(cfg: dict):
    name = str(cfg["model"].get("dtype", "float16")).lower()

    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(name, torch.float16)


def load_source_dataset(cfg: dict):
    local_path = cfg.get("dataset", {}).get("local_path")

    if local_path:
        ds = load_from_disk(resolve_project_path(local_path))
        split = cfg["dataset"].get("split")

        if hasattr(ds, "keys"):
            if split and split in ds:
                ds = ds[split]
            elif "train" in ds:
                ds = ds["train"]
            else:
                ds = ds[list(ds.keys())[0]]
    else:
        ds = load_dataset(
            cfg["dataset"]["name"],
            split=cfg["dataset"].get("split", "train"),
            cache_dir=str(dataset_cache_dir()),
        )

    limit = cfg["dataset"].get("limit")
    if limit is not None:
        ds = ds.select(range(min(int(limit), len(ds))), keep_in_memory=True)

    return ds


def load_tokenizer_cached(cfg: dict):
    tok = AutoTokenizer.from_pretrained(
        model_path_from_cfg(cfg),
        trust_remote_code=True,
        cache_dir=str(hf_home()),
        local_files_only=bool(cfg.get("runtime", {}).get("offline", True)),
        use_fast=True,
    )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return tok


def load_model_cached(cfg: dict):
    model = AutoModelForCausalLM.from_pretrained(
        model_path_from_cfg(cfg),
        torch_dtype=dtype_from_cfg(cfg),
        device_map="auto",
        trust_remote_code=True,
        cache_dir=str(hf_home()),
        local_files_only=bool(cfg.get("runtime", {}).get("offline", True)),
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def chunk_bounds(n_rows: int, chunk_size: int, task_id: int):
    start = int(task_id) * int(chunk_size)
    end = min(start + int(chunk_size), int(n_rows))

    if start >= n_rows:
        return None, None

    return start, end


def num_chunks(n_rows: int, chunk_size: int) -> int:
    return (int(n_rows) + int(chunk_size) - 1) // int(chunk_size)


def atomic_write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp.replace(path)


def atomic_to_parquet(df, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def require_columns(dataset, columns: list[str]) -> None:
    missing = [c for c in columns if c not in dataset.column_names]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}. Existing columns: {dataset.column_names}")
