#!/usr/bin/env python3
"""
Container-side training script for pythoncodes CL experiments.

Supports:
  - plain SFT
  - staged curriculum: easy -> medium -> hard
  - cumulative curriculum: easy -> easy+medium -> easy+medium+hard
  - distribution curriculum: 80/15/5 -> 40/40/20 -> 20/20/60

Smoke-test env vars:
  TRAIN_MAX_STEPS_OVERRIDE=2
  TRAIN_DATASET_LIMIT_OVERRIDE=900
  TRAIN_VAL_SIZE_OVERRIDE=100
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Any

import unsloth  # noqa: F401  # must be imported before transformers/trl/peft
from unsloth import FastLanguageModel

import pandas as pd
import torch
import yaml
from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import snapshot_download
try:
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

from transformers import DataCollatorForSeq2Seq, TrainerCallback, TrainingArguments, set_seed

try:
    from trl import SFTConfig, SFTTrainer
except ImportError:
    SFTConfig = None
    from trl import SFTTrainer

import src.config as config
from src.data.prompt import pythoncodes_cl_scored


LOG_ROOT = config.LOGS_DIR / "train"
RUN_ROOT = config.OUTPUTS_DIR / "train_runs"
TB_ROOT = config.OUTPUTS_DIR / "runs" / "train"


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


class JsonlMetricCallback(TrainerCallback):
    def __init__(self, jsonl_path: Path, logger: logging.Logger, tb_dir: Path | None = None):
        self.jsonl_path = jsonl_path
        self.logger = logger
        self.latest_grad_norm: float | None = None
        self.writer = SummaryWriter(log_dir=str(tb_dir)) if tb_dir is not None else None
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        logging_steps = max(1, int(getattr(args, "logging_steps", 1) or 1))
        if int(state.global_step) % logging_steps != 0:
            return

        total_sq = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            grad = p.grad.detach().float()
            param_norm = grad.norm(2).item()
            total_sq += param_norm * param_norm
        self.latest_grad_norm = float(total_sq ** 0.5)

        if self.writer is not None:
            self.writer.add_scalar("train/manual_grad_norm", self.latest_grad_norm, int(state.global_step))
            self.writer.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        payload = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
            **{k: _json_safe(v) for k, v in logs.items()},
        }

        if "grad_norm" not in payload and self.latest_grad_norm is not None:
            payload["grad_norm"] = self.latest_grad_norm

        if "eval_loss" in payload:
            try:
                payload["eval_perplexity"] = float(math.exp(min(float(payload["eval_loss"]), 20.0)))
            except Exception:
                pass

        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        short = ", ".join(
            f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in payload.items()
            if k not in {"time"}
        )
        self.logger.info(short)

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def _int_env(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def make_logger(run_name: str) -> logging.Logger:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"train.{run_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(LOG_ROOT / f"{run_name}.log", encoding="utf-8", mode="a")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_like: str | None) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like)
    if not path.is_absolute():
        path = config.PROJECT_DIR / path
    return path


def ensure_base_model(local_files_only: bool = False) -> None:
    model_dir = Path(config.MODEL_PATH)
    if (model_dir / "config.json").exists():
        return

    snapshot_download(
        repo_id=config.MODEL_NAME,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        local_files_only=local_files_only,
    )


def load_scored_dataset(cfg: dict[str, Any]) -> Dataset:
    ds_cfg = cfg["dataset"]
    disk_path = resolve_path(ds_cfg.get("disk_path"))
    parquet_path = resolve_path(ds_cfg.get("parquet_path"))

    if parquet_path and parquet_path.exists():
        dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
    elif disk_path and disk_path.exists():
        dataset = load_from_disk(str(disk_path))
        if hasattr(dataset, "keys"):
            dataset = dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    else:
        raise FileNotFoundError(
            f"Cannot find scored dataset. Checked disk_path={disk_path}, parquet_path={parquet_path}"
        )

    limit = _int_env("TRAIN_DATASET_LIMIT_OVERRIDE", ds_cfg.get("limit"))
    if limit:
        dataset = dataset.select(range(min(int(limit), len(dataset))), keep_in_memory=True)

    return dataset


def split_dataset(dataset: Dataset, cfg: dict[str, Any]) -> tuple[Dataset, Dataset]:
    ds_cfg = cfg["dataset"]
    seed = int(ds_cfg.get("seed", 42))
    val_size = _int_env("TRAIN_VAL_SIZE_OVERRIDE", int(ds_cfg.get("val_size", 1000)))
    assert val_size is not None

    dataset = dataset.shuffle(seed=seed, keep_in_memory=True)
    val_size = min(int(val_size), max(1, len(dataset) // 10), len(dataset) - 1)
    val_dataset = dataset.select(range(val_size), keep_in_memory=True)
    train_dataset = dataset.select(range(val_size, len(dataset)), keep_in_memory=True)
    return train_dataset, val_dataset


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _dataset_select_by_indices(dataset: Dataset, indices: list[int]) -> Dataset:
    if not indices:
        return dataset.select([], keep_in_memory=True)
    return dataset.select(indices, keep_in_memory=True)


def filter_stage(dataset: Dataset, category_col: str, wanted: set[str]) -> Dataset:
    if category_col not in dataset.column_names:
        raise KeyError(f"Column {category_col!r} not found. Available: {dataset.column_names}")
    return dataset.filter(lambda row: row.get(category_col) in wanted, keep_in_memory=True)


def sample_distribution_stage(
    dataset: Dataset,
    category_col: str,
    weights: dict[str, float],
    sample_size: int,
    seed: int,
) -> Dataset:
    if category_col not in dataset.column_names:
        raise KeyError(f"Column {category_col!r} not found. Available: {dataset.column_names}")

    rng = random.Random(seed)
    by_cat: dict[str, list[int]] = {cat: [] for cat in weights}
    for i, value in enumerate(dataset[category_col]):
        if value in by_cat:
            by_cat[value].append(i)

    if not any(by_cat.values()):
        return dataset.select([], keep_in_memory=True)

    # Normalize weights after dropping empty categories.
    active = {cat: w for cat, w in weights.items() if by_cat.get(cat)}
    total_weight = sum(float(w) for w in active.values())
    active = {cat: float(w) / total_weight for cat, w in active.items()}

    indices: list[int] = []
    cats = list(active.keys())
    for cat in cats:
        n = int(round(sample_size * active[cat]))
        pool = by_cat[cat]
        if not pool:
            continue
        # Use replacement if requested stage is larger than available rows.
        if n <= len(pool):
            chosen = rng.sample(pool, n)
        else:
            chosen = [rng.choice(pool) for _ in range(n)]
        indices.extend(chosen)

    # Correct rounding drift.
    while len(indices) < sample_size:
        cat = rng.choices(cats, weights=[active[c] for c in cats], k=1)[0]
        indices.append(rng.choice(by_cat[cat]))
    if len(indices) > sample_size:
        indices = indices[:sample_size]

    rng.shuffle(indices)
    return _dataset_select_by_indices(dataset, indices)


def build_stage_specs(run_cfg: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    category_col = run_cfg.get("category_col")
    schedule_type = str(run_cfg.get("schedule_type") or ("plain" if not category_col else "staged"))
    stages = list(cfg.get("curriculum", {}).get("stages", ["easy", "medium", "hard"]))

    if schedule_type == "plain" or not category_col:
        return [{"name": "all", "categories": ["all"], "schedule_type": "plain"}]

    if schedule_type == "staged":
        return [
            {"name": stage, "categories": [stage], "schedule_type": "staged"}
            for stage in stages
        ]

    if schedule_type == "cumulative":
        specs = []
        for i, stage in enumerate(stages):
            specs.append({
                "name": "+".join(stages[: i + 1]),
                "categories": stages[: i + 1],
                "schedule_type": "cumulative",
            })
        return specs

    if schedule_type == "distribution":
        specs = []
        for item in cfg.get("curriculum", {}).get("distribution_stages", []):
            specs.append({
                "name": str(item["name"]),
                "weights": dict(item["weights"]),
                "schedule_type": "distribution",
            })
        return specs

    raise ValueError(f"Unknown schedule_type={schedule_type!r} for run={run_cfg.get('name')}")


def get_stage_raw_dataset(
    full_train_dataset: Dataset,
    run_cfg: dict[str, Any],
    cfg: dict[str, Any],
    stage_spec: dict[str, Any],
    seed: int,
) -> Dataset:
    schedule_type = stage_spec["schedule_type"]
    category_col = run_cfg.get("category_col")

    if schedule_type == "plain" or not category_col:
        return full_train_dataset

    if schedule_type in {"staged", "cumulative"}:
        return filter_stage(full_train_dataset, category_col, set(stage_spec["categories"]))

    if schedule_type == "distribution":
        sample_size = int(cfg.get("curriculum", {}).get("distribution_stage_size") or len(full_train_dataset))
        sample_size = min(sample_size, len(full_train_dataset))
        return sample_distribution_stage(
            full_train_dataset,
            category_col=category_col,
            weights=stage_spec["weights"],
            sample_size=sample_size,
            seed=seed,
        )

    raise ValueError(f"Unknown schedule_type={schedule_type!r}")


def prepare_tokenized_dataset(dataset: Dataset, tokenizer, train: bool, seed: int) -> Dataset:
    mapped = dataset.map(
        pythoncodes_cl_scored.build_tokenized_chat,
        fn_kwargs={"tokenizer": tokenizer, "max_length": config.MAX_TOKENS, "train": train},
        remove_columns=list(dataset.column_names),
        desc="Tokenizing scored pythoncodes chats",
        keep_in_memory=True,
    )
    if train:
        mapped = mapped.shuffle(seed=seed, keep_in_memory=True)
    return mapped


def make_training_arguments(**kwargs) -> TrainingArguments:
    args_cls = SFTConfig if SFTConfig is not None else TrainingArguments
    params = signature(args_cls.__init__).parameters

    if "eval_strategy" not in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if "eval_strategy" in params and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    if "max_seq_length" in kwargs and "max_length" in params and "max_seq_length" not in params:
        kwargs["max_length"] = kwargs.pop("max_seq_length")
    if "max_length" in kwargs and "max_seq_length" in params and "max_length" not in params:
        kwargs["max_seq_length"] = kwargs.pop("max_length")

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return args_cls(**filtered)


def make_sft_trainer(
    *,
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    args: TrainingArguments,
    callbacks: list[TrainerCallback],
) -> SFTTrainer:
    params = signature(SFTTrainer.__init__).parameters
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": args,
        "callbacks": callbacks,
        "data_collator": DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
    }

    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer

    return SFTTrainer(**trainer_kwargs)


def effective_stage_max_steps(tcfg: dict[str, Any], schedule_type: str) -> int:
    override = _int_env("TRAIN_MAX_STEPS_OVERRIDE", None)
    if override is not None:
        return int(override)
    if schedule_type == "plain":
        return int(tcfg.get("max_steps", 600))
    return int(tcfg.get("stage_max_steps", tcfg.get("max_steps", 200)))


def train_one_run(
    run_cfg: dict[str, Any],
    full_train_dataset: Dataset,
    full_val_dataset: Dataset,
    cfg: dict[str, Any],
    force: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    run_name = run_cfg["name"]
    category_col = run_cfg.get("category_col")
    schedule_type = str(run_cfg.get("schedule_type") or ("plain" if not category_col else "staged"))
    logger = make_logger(run_name)

    model_slug = Path(config.MODEL_PATH).name
    save_path = config.MODELS_DIR / f"{model_slug}-sft-{run_name}"
    run_out = RUN_ROOT / run_name
    tb_run = TB_ROOT / run_name
    metrics_jsonl = run_out / "metrics.jsonl"

    if force and not dry_run:
        shutil.rmtree(save_path, ignore_errors=True)
        shutil.rmtree(run_out, ignore_errors=True)
        shutil.rmtree(tb_run, ignore_errors=True)

    if not dry_run and save_path.exists() and (save_path / "adapter_config.json").exists():
        logger.info("Adapter already exists: %s", save_path)
        return {
            "run_name": run_name,
            "category_col": category_col,
            "schedule_type": schedule_type,
            "adapter_path": str(save_path),
            "status": "skipped_exists",
        }

    run_out.mkdir(parents=True, exist_ok=True)
    tb_run.mkdir(parents=True, exist_ok=True)

    tcfg = cfg["training"]
    lcfg = cfg["lora"]
    mcfg = cfg["model"]
    seed = int(tcfg.get("seed", cfg["dataset"].get("seed", 42)))
    stage_specs = build_stage_specs(run_cfg, cfg)

    if dry_run:
        logger.info("DRY RUN: run=%s schedule_type=%s category_col=%s", run_name, schedule_type, category_col)
        rows = []
        for stage_idx, stage_spec in enumerate(stage_specs, start=1):
            stage_raw = get_stage_raw_dataset(
                full_train_dataset, run_cfg, cfg, stage_spec, seed=seed + stage_idx
            )
            rows.append({
                "stage_idx": stage_idx,
                "stage_name": stage_spec["name"],
                "schedule_type": stage_spec["schedule_type"],
                "categories": ",".join(stage_spec.get("categories", [])),
                "weights": json.dumps(stage_spec.get("weights", {}), ensure_ascii=False),
                "rows": len(stage_raw),
                "max_steps": effective_stage_max_steps(tcfg, stage_spec["schedule_type"]),
            })
        for row in rows:
            logger.info("DRY STAGE: %s", row)
        (run_out / "dry_run_stages.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "run_name": run_name,
            "category_col": category_col,
            "schedule_type": schedule_type,
            "adapter_path": str(save_path),
            "status": "dry_run",
        }

    with tee_to_file(LOG_ROOT / f"{run_name}.stdout.log"):
        logger.info("Starting run=%s category_col=%s schedule_type=%s", run_name, category_col, schedule_type)
        logger.info("Base model: %s", config.MODEL_PATH)
        logger.info("Output adapter: %s", save_path)
        logger.info("TRAIN_MAX_STEPS_OVERRIDE=%s", os.environ.get("TRAIN_MAX_STEPS_OVERRIDE", ""))
        logger.info("TRAIN_DATASET_LIMIT_OVERRIDE=%s", os.environ.get("TRAIN_DATASET_LIMIT_OVERRIDE", ""))
        logger.info("TRAIN_VAL_SIZE_OVERRIDE=%s", os.environ.get("TRAIN_VAL_SIZE_OVERRIDE", ""))

        set_seed(seed)
        random.seed(seed)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.MODEL_PATH,
            max_seq_length=config.MAX_TOKENS,
            dtype=mcfg.get("dtype"),
            load_in_4bit=bool(mcfg.get("load_in_4bit", True)),
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lcfg.get("r", 32)),
            target_modules=list(lcfg["target_modules"]),
            lora_alpha=int(lcfg.get("alpha", 32)),
            lora_dropout=float(lcfg.get("dropout", 0)),
            bias=str(lcfg.get("bias", "none")),
            use_gradient_checkpointing=lcfg.get("use_gradient_checkpointing", "unsloth"),
        )

        val_tokenized = prepare_tokenized_dataset(full_val_dataset, tokenizer, train=True, seed=seed)

        final_train_result = None
        stage_summary: list[dict[str, Any]] = []
        for stage_idx, stage_spec in enumerate(stage_specs, start=1):
            stage_raw = get_stage_raw_dataset(full_train_dataset, run_cfg, cfg, stage_spec, seed=seed + stage_idx)
            if len(stage_raw) == 0:
                logger.warning("Stage %s skipped: empty dataset", stage_spec["name"])
                continue

            train_tokenized = prepare_tokenized_dataset(stage_raw, tokenizer, train=True, seed=seed + stage_idx)
            safe_stage_name = str(stage_spec["name"]).replace("+", "plus").replace("/", "_")
            stage_name = f"{run_name}_stage_{stage_idx}_{safe_stage_name}"
            stage_dir = run_out / stage_name
            stage_tb = tb_run / stage_name
            max_steps = effective_stage_max_steps(tcfg, stage_spec["schedule_type"])

            logger.info(
                "Stage %s/%s: %s | schedule=%s | categories=%s | weights=%s | rows=%s | val_rows=%s | max_steps=%s",
                stage_idx,
                len(stage_specs),
                stage_spec["name"],
                stage_spec["schedule_type"],
                stage_spec.get("categories"),
                stage_spec.get("weights"),
                len(train_tokenized),
                len(val_tokenized),
                max_steps,
            )

            args = make_training_arguments(
                output_dir=str(stage_dir / "checkpoints"),
                run_name=stage_name,
                per_device_train_batch_size=int(tcfg.get("train_batch_size", 1)),
                gradient_accumulation_steps=int(tcfg.get("gradient_steps", 8)),
                max_steps=int(max_steps),
                warmup_steps=min(int(tcfg.get("warmup_steps", 25)), max(0, int(max_steps) - 1)),
                learning_rate=float(tcfg.get("lr", 2e-4)),
                lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),
                optim=str(tcfg.get("optim", "paged_adamw_8bit")),
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_strategy="steps",
                logging_steps=max(1, min(int(tcfg.get("logging_steps", 10)), int(max_steps))),
                eval_strategy="steps",
                eval_steps=max(1, min(int(tcfg.get("eval_steps", 50)), int(max_steps))),
                save_strategy="steps",
                save_steps=max(1, min(int(tcfg.get("save_steps", 100)), int(max_steps))),
                save_total_limit=int(tcfg.get("save_total_limit", 2)),
                report_to=[str(tcfg.get("report_to", "tensorboard"))],
                logging_dir=str(stage_tb),
                dataloader_num_workers=int(tcfg.get("dataloader_num_workers", 2)),
                dataloader_pin_memory=bool(tcfg.get("dataloader_pin_memory", True)),
                max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
                seed=seed,
                remove_unused_columns=False,
                max_seq_length=config.MAX_TOKENS,
                max_length=config.MAX_TOKENS,
                packing=False,
                assistant_only_loss=False,  # labels already contain assistant-only mask
                dataset_kwargs={"skip_prepare_dataset": True},
            )

            trainer = make_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_tokenized,
                eval_dataset=val_tokenized,
                args=args,
                callbacks=[JsonlMetricCallback(metrics_jsonl, logger, tb_dir=stage_tb)],
            )

            final_train_result = trainer.train()
            eval_metrics = trainer.evaluate()
            logger.info("Stage %s final eval: %s", stage_spec["name"], eval_metrics)

            eval_path = run_out / f"{stage_name}_final_eval.json"
            eval_payload = {k: _json_safe(v) for k, v in eval_metrics.items()}
            eval_path.write_text(json.dumps(eval_payload, indent=2, ensure_ascii=False), encoding="utf-8")

            stage_summary.append({
                "run_name": run_name,
                "stage_idx": stage_idx,
                "stage_name": stage_spec["name"],
                "schedule_type": stage_spec["schedule_type"],
                "categories": ",".join(stage_spec.get("categories", [])),
                "weights": json.dumps(stage_spec.get("weights", {}), ensure_ascii=False),
                "rows": len(train_tokenized),
                "max_steps": max_steps,
                **eval_payload,
            })

            del trainer, train_tokenized, stage_raw
            cleanup_cuda()

        logger.info("Saving LoRA adapter: %s", save_path)
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))

        if stage_summary:
            pd.DataFrame(stage_summary).to_csv(run_out / "stage_summary.csv", index=False)
            pd.DataFrame(stage_summary).to_markdown(run_out / "stage_summary.md", index=False)
            (run_out / "stage_summary.json").write_text(
                json.dumps(stage_summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        del model, tokenizer, val_tokenized
        cleanup_cuda()

    status = "ok" if final_train_result is not None else "no_non_empty_stages"
    return {
        "run_name": run_name,
        "category_col": category_col,
        "schedule_type": schedule_type,
        "adapter_path": str(save_path),
        "metrics_jsonl": str(metrics_jsonl),
        "tensorboard_dir": str(tb_run),
        "status": status,
    }


def print_run_table(cfg: dict[str, Any]) -> None:
    rows = []
    for i, run_cfg in enumerate(cfg["runs"]):
        rows.append({
            "idx": i,
            "name": run_cfg["name"],
            "schedule_type": run_cfg.get("schedule_type"),
            "category_col": run_cfg.get("category_col"),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/lora_pythoncodes_cl.yaml")
    parser.add_argument("--only", nargs="*", default=None, help="Optional run names")
    parser.add_argument("--force", action="store_true", help="Delete existing adapters/logs for selected runs")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--list-runs", action="store_true", help="Print configured runs and exit")
    parser.add_argument("--dry-run", action="store_true", help="Build stage plans without loading model/training")
    args = parser.parse_args()

    cfg_path = resolve_path(args.config)
    assert cfg_path is not None
    cfg = read_yaml(cfg_path)

    if args.list_runs:
        print_run_table(cfg)
        return 0

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    TB_ROOT.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        ensure_base_model(local_files_only=args.local_files_only)

    dataset = load_scored_dataset(cfg)
    train_dataset, val_dataset = split_dataset(dataset, cfg)
    print(f"Loaded scored dataset: train={len(train_dataset)}, val={len(val_dataset)}", flush=True)

    selected = set(args.only) if args.only else None
    results = []
    for run_cfg in cfg["runs"]:
        if selected and run_cfg["name"] not in selected:
            continue
        cleanup_cuda()
        result = train_one_run(
            run_cfg,
            train_dataset,
            val_dataset,
            cfg,
            force=args.force,
            dry_run=args.dry_run,
        )
        results.append(result)

    manifest = pd.DataFrame(results)
    manifests_dir = RUN_ROOT / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # In Slurm array mode each task trains exactly one run. Avoid concurrent writes
    # to one shared trained_adapters.csv; write one small manifest per task instead.
    cluster_mode = os.environ.get("CLUSTER_ARRAY_MODE", "") == "1"
    if cluster_mode or (selected and len(results) == 1):
        for result in results:
            run_name = str(result.get("run_name", "unknown"))
            per_run = pd.DataFrame([result])
            per_csv = manifests_dir / f"{run_name}.csv"
            per_json = manifests_dir / f"{run_name}.json"
            per_csv.parent.mkdir(parents=True, exist_ok=True)
            per_run.to_csv(per_csv, index=False)
            per_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved per-run training manifest: {per_csv}")
    else:
        manifest_path = RUN_ROOT / "trained_adapters.csv"
        manifest.to_csv(manifest_path, index=False)
        manifest.to_markdown(RUN_ROOT / "trained_adapters.md", index=False)
        print(f"Saved training manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
