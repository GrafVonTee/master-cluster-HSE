#!/usr/bin/env python3
"""
Evaluate clean base model and all saved LoRA adapters sequentially in one eval container.
Writes raw summary plus comparison tables with deltas against base and ordinary SFT.
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import datasets
import pandas as pd
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def add_text(self, *args, **kwargs):
            pass
        def add_hparams(self, *args, **kwargs):
            pass
        def flush(self):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
            return False

except Exception:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def add_text(self, *args, **kwargs):
            pass
        def add_hparams(self, *args, **kwargs):
            pass
        def flush(self):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
            return False


import src.config as config
from src.data.loader import load_benchmark
from src.data.prompt import humaneval, mbpp
from src.evaluator import Evaluator
from src.inference.vllm_inference import setup_model
from src.metrics import GreedyPass, MeanEntropy, PassAtk, PercentPassed

try:
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
except Exception:
    destroy_distributed_environment = None
    destroy_model_parallel = None


LOG_ROOT = config.LOGS_DIR / "eval"
EVAL_ROOT = config.OUTPUTS_DIR / "eval"
TB_ROOT = config.OUTPUTS_DIR / "runs" / "eval"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_to_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, f)
        sys.stderr = Tee(old_stderr, f)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def make_logger(name: str) -> logging.Logger:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"eval.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(LOG_ROOT / f"{name}.log", encoding="utf-8", mode="a")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def cleanup_vllm() -> None:
    try:
        if destroy_model_parallel:
            destroy_model_parallel()
    except Exception:
        pass
    try:
        if destroy_distributed_environment:
            destroy_distributed_environment()
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def prepare_experiments(only: set[str] | None = None, include_base: bool = False) -> list[tuple[str, str, str | None]]:
    experiments: list[tuple[str, str, str | None]] = []
    model_slug = Path(config.MODEL_PATH).name
    expected_prefix = f"{model_slug}-sft-"

    if include_base and (only is None or "base" in only):
        experiments.append(("base", str(config.MODEL_PATH), None))

    if not config.MODELS_DIR.exists():
        return experiments

    for folder_name in sorted(os.listdir(config.MODELS_DIR)):
        full_path = config.MODELS_DIR / folder_name
        if not full_path.is_dir() or folder_name.startswith("."):
            continue
        if folder_name == model_slug:
            continue
        if not folder_name.startswith(expected_prefix):
            continue

        exp_name = folder_name.removeprefix(expected_prefix)
        if only and exp_name not in only and folder_name not in only:
            continue
        if not (full_path / "adapter_config.json").exists():
            continue
        experiments.append((exp_name, str(config.MODEL_PATH), str(full_path)))

    order = {
        "base": 0,
        "sft_pythoncodes": 1,
        "cl_length": 2,
        "cl_perplexity": 3,
        "cl_lexical": 4,
        "cl_semantic": 5,
        "cl_llm_judge": 6,
    }
    return sorted(experiments, key=lambda x: (order.get(x[0], 99), x[0]))


def load_raw_benchmark(benchmark: str):
    import datasets
    from pathlib import Path

    name = str(benchmark).lower().strip()
    local_root = Path("/workspace/datasets")

    if name == "mbpp":
        local_path = local_root / "mbpp"
        if local_path.exists():
            return datasets.load_from_disk(str(local_path))
        return datasets.load_dataset("google-research-datasets/mbpp")

    if name in {"humaneval", "human_eval", "openai_humaneval"}:
        local_path = local_root / "humaneval"
        if local_path.exists():
            return datasets.load_from_disk(str(local_path))
        return datasets.load_dataset("openai/openai_humaneval")

    raise ValueError(f"Unknown benchmark: {benchmark!r}")

def build_tasks(benchmark: str, raw_dataset, tokenizer, limit: int | None):
    if benchmark == "mbpp":
        mapper = functools.partial(mbpp.mbpp_to_task, tokenizer=tokenizer)
    elif benchmark == "humaneval":
        mapper = functools.partial(humaneval.humaneval_to_task, tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    tasks = load_benchmark(raw_dataset, mapper)
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def _json_safe_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def write_comparison_tables(df: pd.DataFrame) -> None:
    if df.empty:
        return

    metric_cols = [
        c for c in ["greedy@1", "pass@1 (n=1)", "mean_%passed", "mean_entropy"]
        if c in df.columns
    ]
    long_rows: list[dict[str, Any]] = []

    for benchmark, part in df.groupby("benchmark", dropna=False):
        base_rows = part[part["experiment"] == "base"]
        sft_rows = part[part["experiment"] == "sft_pythoncodes"]
        base = base_rows.iloc[0] if len(base_rows) else None
        sft = sft_rows.iloc[0] if len(sft_rows) else None

        for _, row in part.iterrows():
            out = {
                "benchmark": benchmark,
                "experiment": row["experiment"],
                "num_tasks": row.get("num_tasks", ""),
            }
            for metric in metric_cols:
                value = row.get(metric)
                out[metric] = value
                if base is not None and pd.notna(value) and pd.notna(base.get(metric)):
                    out[f"{metric}_delta_vs_base"] = float(value) - float(base[metric])
                if sft is not None and pd.notna(value) and pd.notna(sft.get(metric)):
                    out[f"{metric}_delta_vs_sft"] = float(value) - float(sft[metric])
            long_rows.append(out)

    cmp_df = pd.DataFrame(long_rows)
    cmp_df.to_csv(EVAL_ROOT / "lora_eval_comparison.csv", index=False)
    cmp_df.to_markdown(EVAL_ROOT / "lora_eval_comparison.md", index=False)

    for metric in metric_cols:
        pivot = df.pivot_table(index="experiment", columns="benchmark", values=metric, aggfunc="first")
        pivot = pivot.reset_index()
        safe_metric = metric.replace("@", "at").replace("%", "pct").replace(" ", "_").replace("(", "").replace(")", "")
        pivot.to_csv(EVAL_ROOT / f"pivot_{safe_metric}.csv", index=False)
        pivot.to_markdown(EVAL_ROOT / f"pivot_{safe_metric}.md", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", default="mbpp,humaneval")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-base", action="store_true")
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    TB_ROOT.mkdir(parents=True, exist_ok=True)

    only = set(args.only) if args.only else None
    experiments = prepare_experiments(only=only, include_base=args.include_base)

    if not experiments:
        raise RuntimeError(
            f"No experiments found in {config.MODELS_DIR}. "
            f"Expected base and/or folders like {Path(config.MODEL_PATH).name}-sft-cl_length"
        )

    benchmarks = [b.strip().lower() for b in args.benchmarks.split(",") if b.strip()]
    metrics = [
        GreedyPass(),
        PassAtk(k=1, n_samples=1),
        PercentPassed(),
        MeanEntropy(),
    ]

    summary_rows: list[dict[str, Any]] = []
    jsonl_path = EVAL_ROOT / "lora_eval_results.jsonl"
    writer = SummaryWriter(log_dir=str(TB_ROOT))

    try:
        for exp_idx, (exp_name, model_path, adapter_path) in enumerate(experiments, start=1):
            logger = make_logger(exp_name)
            with tee_to_file(LOG_ROOT / f"{exp_name}.stdout.log"):
                logger.info("Running experiment %s/%s: %s", exp_idx, len(experiments), exp_name)
                logger.info("model_path=%s adapter_path=%s", model_path, adapter_path)

                llm, tokenizer, _ = setup_model(model_path=model_path, adapter_path=adapter_path)
                evaluator = Evaluator(llm, tokenizer, metrics)

                for benchmark in benchmarks:
                    logger.info("Benchmark: %s", benchmark)
                    raw = load_raw_benchmark(benchmark)
                    tasks = build_tasks(benchmark, raw, tokenizer, args.limit)
                    logger.info("Tasks: %s", len(tasks))

                    result = evaluator.run(tasks)
                    row = {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "experiment": exp_name,
                        "model_path": model_path,
                        "adapter_path": adapter_path or "",
                        "benchmark": benchmark,
                        "num_tasks": len(tasks),
                        **result,
                    }
                    row = {k: _json_safe_value(v) for k, v in row.items()}
                    summary_rows.append(row)

                    with jsonl_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

                    for metric_name, value in result.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(f"{benchmark}/{metric_name}", float(value), exp_idx)
                            writer.add_text(f"{benchmark}/{metric_name}_label", exp_name, exp_idx)

                    logger.info("Result row: %s", row)

                del evaluator, llm, tokenizer
                cleanup_vllm()
    finally:
        writer.flush()
        writer.close()

    df = pd.DataFrame(summary_rows)
    csv_path = EVAL_ROOT / "lora_eval_summary.csv"
    md_path = EVAL_ROOT / "lora_eval_summary.md"
    json_path = EVAL_ROOT / "lora_eval_summary.json"
    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    write_comparison_tables(df)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved comparison: {EVAL_ROOT / 'lora_eval_comparison.md'}")
    print(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
