from __future__ import annotations

import inspect
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML dict in {path}, got {type(data).__name__}")
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    # Keep torch optional at module import time. Local smoke utilities that only need
    # read_yaml/write_json must work in lightweight venvs without PyTorch installed.
    try:
        import torch
    except ModuleNotFoundError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def env_override_int(name: str, current: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return current
    return int(raw)


def env_override_float(name: str, current: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return current
    return float(raw)


def env_override_str(name: str, current: str | None) -> str | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return current
    return str(raw)


def filter_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
    except Exception:
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
