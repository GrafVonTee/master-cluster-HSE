import gc
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

import src.config as config
from src.data.loader import load_benchmark
from src.data.prompt import humaneval, mbpp
from src.executor import LocalExecutor
from src.inference.vllm_inference import make_sampling_params, setup_model
from src.metrics import GreedyPass, MeanEntropy, PassAtk, PercentPassed


def cleanup_cuda():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def cleanup_vllm():
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass

    cleanup_cuda()


def _select_split(dataset, split: str = "test"):
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"В DatasetDict нет split='{split}'. Доступно: {list(dataset)}")

        return dataset[split]

    return dataset


def _load_local_dataset(candidates: Iterable[str], split: str = "test"):
    checked = []

    for name in candidates:
        path = Path(config.DATASETS_DIR) / name
        checked.append(path)

        if path.exists():
            dataset = load_from_disk(str(path))
            return _select_split(dataset, split=split), path

    return None, checked


def _download_dataset(hub_path: str, hub_name: Optional[str] = None, split: str = "test"):
    if hub_name is None:
        return load_dataset(hub_path, split=split)

    return load_dataset(hub_path, hub_name, split=split)


def prepare_eval_datasets(overwrite: bool = False):
    data_dir = Path(config.DATASETS_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        (
            data_dir / "mbpp_sanitized_test",
            "google-research-datasets/mbpp",
            "sanitized",
            "test",
        ),
        (
            data_dir / "humaneval_test",
            "openai/openai_humaneval",
            None,
            "test",
        ),
    ]

    rows = []

    for out_path, hub_path, hub_name, split in specs:
        if out_path.exists() and not overwrite:
            dataset = load_from_disk(str(out_path))
            rows.append(
                {
                    "path": str(out_path),
                    "rows": len(dataset),
                    "status": "exists",
                }
            )
            continue

        dataset = _download_dataset(hub_path, hub_name=hub_name, split=split)
        dataset.save_to_disk(str(out_path))

        rows.append(
            {
                "path": str(out_path),
                "rows": len(dataset),
                "status": "saved",
            }
        )

    return pd.DataFrame(rows)


def load_eval_benchmarks(allow_download: bool = False):
    mbpp_ds, mbpp_checked = _load_local_dataset(
        [
            "mbpp_sanitized_test",
            "mbpp_sanitized",
            "mbpp",
        ],
        split="test",
    )

    if mbpp_ds is None:
        if not allow_download:
            raise FileNotFoundError(
                "Не найден локальный MBPP eval dataset. Проверены пути:\n"
                + "\n".join(f"  {p}" for p in mbpp_checked)
                + "\n\nОдин раз выполни prepare_eval_datasets() с интернетом."
            )

        mbpp_ds = _download_dataset(
            "google-research-datasets/mbpp",
            hub_name="sanitized",
            split="test",
        )

    he_ds, he_checked = _load_local_dataset(
        [
            "humaneval_test",
            "openai_humaneval_test",
            "humaneval",
            "openai_humaneval",
        ],
        split="test",
    )

    if he_ds is None:
        if not allow_download:
            raise FileNotFoundError(
                "Не найден локальный HumanEval eval dataset. Проверены пути:\n"
                + "\n".join(f"  {p}" for p in he_checked)
                + "\n\nОдин раз выполни prepare_eval_datasets() с интернетом."
            )

        he_ds = _download_dataset("openai/openai_humaneval", split="test")

    return [
        ("mbpp", mbpp_ds, mbpp.mbpp_to_task),
        ("humaneval", he_ds, humaneval.humaneval_to_task),
    ]


def _metric_groups(metrics):
    groups = {}

    for metric in metrics:
        key = json.dumps(metric.gen_config, sort_keys=True)

        if key not in groups:
            groups[key] = {
                "generation_config": dict(metric.gen_config),
                "metrics": [],
            }

        groups[key]["metrics"].append(metric)

    return list(groups.values())


def _calculate_entropy(sample_output) -> float:
    if not getattr(sample_output, "logprobs", None):
        return 0.0

    entropies = []

    for step_logprobs in sample_output.logprobs:
        if not step_logprobs:
            continue

        value = list(step_logprobs.values())[0].logprob
        entropies.append(-value)

    return float(np.mean(entropies)) if entropies else 0.0


def _run_eval_tasks(llm, tokenizer, tasks, metrics):
    executor = LocalExecutor()
    final_results = {}

    for group in _metric_groups(metrics):
        sampling_params = make_sampling_params(
            tokenizer,
            overrides=group["generation_config"],
        )
        metrics_in_group = group["metrics"]

        print(f"\nGroup: {[m.name for m in metrics_in_group]}")
        print(
            "Params: "
            f"n={sampling_params.n}, "
            f"temp={sampling_params.temperature}, "
            f"max_tokens={sampling_params.max_tokens}, "
            f"logprobs={sampling_params.logprobs}"
        )

        prompts = [task.prompt for task in tasks]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        flat_tasks_input = []
        map_indices = []

        for task_idx, request_output in enumerate(outputs):
            task_data = tasks[task_idx]

            for sample_idx, sample in enumerate(request_output.outputs):
                flat_tasks_input.append((sample.text, task_data.tests))
                map_indices.append((task_idx, sample_idx))

        flat_results = executor.batch_execute(flat_tasks_input)

        all_exec_results = [[] for _ in range(len(tasks))]

        for flat_idx, exec_res in enumerate(flat_results):
            task_idx, sample_idx = map_indices[flat_idx]
            original_sample = outputs[task_idx].outputs[sample_idx]
            exec_res.entropy = _calculate_entropy(original_sample)
            all_exec_results[task_idx].append(exec_res)

        for metric in metrics_in_group:
            score = metric.calculate(all_exec_results)
            final_results[metric.name] = score
            print(f"{metric.name}: {score:.4f}")

    return final_results


def _limit_dataset(dataset: Dataset, max_tasks: Optional[int]):
    if max_tasks is None:
        return dataset

    return dataset.select(range(min(int(max_tasks), len(dataset))))


def evaluate_lora_experiments(
    experiments,
    max_tasks: Optional[int] = None,
    out_name: str = "cl_eval",
    allow_download: bool = False,
    benchmarks: Optional[Iterable[str]] = None,
):
    wanted_benchmarks = set(benchmarks) if benchmarks is not None else None
    raw_benchmarks = load_eval_benchmarks(allow_download=allow_download)

    metrics = [
        GreedyPass(),
        PassAtk(k=1, n_samples=1),
        PercentPassed(),
        MeanEntropy(),
    ]

    records = []

    for i, (exp_name, model_path, adapter_path) in enumerate(experiments):
        print(f"=== Eval {i + 1}/{len(experiments)}: {exp_name} ===")

        llm = None
        tokenizer = None

        try:
            llm, tokenizer, _ = setup_model(
                model_path=model_path,
                adapter_path=adapter_path,
            )

            for benchmark, raw, mapper_fn in raw_benchmarks:
                if wanted_benchmarks is not None and benchmark not in wanted_benchmarks:
                    continue

                raw_part = _limit_dataset(raw, max_tasks=max_tasks)

                def mapper(row, fn=mapper_fn):
                    return fn(row, tokenizer=tokenizer)

                tasks = load_benchmark(raw_part, mapper)

                print(f"Benchmark={benchmark}; tasks={len(tasks)}")
                result = _run_eval_tasks(llm, tokenizer, tasks, metrics)

                records.append(
                    {
                        "experiment": exp_name,
                        "benchmark": benchmark,
                        "num_tasks": len(tasks),
                        "model_path": model_path,
                        "adapter_path": adapter_path,
                        **result,
                    }
                )

        finally:
            if llm is not None:
                del llm

            if tokenizer is not None:
                del tokenizer

            cleanup_vllm()

    out_dir = Path(config.OUTPUTS_DIR) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)

    csv_path = out_dir / f"{out_name}.csv"
    parquet_path = out_dir / f"{out_name}.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    print(f"saved: {csv_path}")
    print(f"saved: {parquet_path}")

    return df
