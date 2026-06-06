#!/usr/bin/env python3
"""Train Qwen/Code models on pythoncodes with Unsloth + TRL GRPO.

This script is intentionally self-contained and conservative. It is designed to
run inside the verified V100 sandbox where Unsloth, vLLM and TRL already work.
"""
from __future__ import annotations

# Unsloth must be imported before transformers/trl for patching.
import unsloth  # noqa: F401

# Unsloth may try to fetch statistics from HuggingFace even for local models.
import os
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

from unsloth import FastLanguageModel

import argparse
import inspect
from pathlib import Path
from typing import Any

from trl import GRPOConfig, GRPOTrainer

from src.rl.grpo_utils import (
    env_override_float,
    env_override_int,
    env_override_str,
    filter_kwargs,
    read_yaml,
    set_seed,
    write_json,
)
from src.rl.pythoncodes_dataset import prepare_pythoncodes_grpo_dataset
from src.rl.rewards import PythonRewardConfig, make_python_reward


def _model_kwargs(model_cfg: dict[str, Any], lora_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        "model_name": model_cfg["base_model"],
        "max_seq_length": int(model_cfg.get("max_seq_length", 2048)),
        "load_in_4bit": bool(model_cfg.get("load_in_4bit", False)),
        "fast_inference": bool(model_cfg.get("fast_inference", True)),
        "max_lora_rank": int(lora_cfg.get("r", 32)),
        "gpu_memory_utilization": float(model_cfg.get("gpu_memory_utilization", 0.70)),
    }
    if "local_files_only" in inspect.signature(FastLanguageModel.from_pretrained).parameters:
        kwargs["local_files_only"] = bool(model_cfg.get("local_files_only", True))
    return filter_kwargs(FastLanguageModel.from_pretrained, kwargs)


def load_model_tokenizer(cfg: dict[str, Any]):
    model_cfg = cfg["model"]
    lora_cfg = cfg.get("lora", {})

    model, tokenizer = FastLanguageModel.from_pretrained(**_model_kwargs(model_cfg, lora_cfg))

    peft_kwargs = {
        "r": int(lora_cfg.get("r", 32)),
        "target_modules": list(lora_cfg.get("target_modules") or [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ]),
        "lora_alpha": int(lora_cfg.get("alpha", lora_cfg.get("r", 32))),
        "lora_dropout": float(lora_cfg.get("dropout", 0.0)),
        "bias": str(lora_cfg.get("bias", "none")),
        "use_gradient_checkpointing": lora_cfg.get("use_gradient_checkpointing", "unsloth"),
        "random_state": int(cfg.get("run", {}).get("seed", 42)),
    }
    peft_kwargs = filter_kwargs(FastLanguageModel.get_peft_model, peft_kwargs)
    model = FastLanguageModel.get_peft_model(model, **peft_kwargs)
    return model, tokenizer


def build_training_args(cfg: dict[str, Any], run_name: str) -> GRPOConfig:
    train_cfg = dict(cfg.get("training", {}))
    model_cfg = cfg.get("model", {})

    train_cfg["max_steps"] = env_override_int("RL_MAX_STEPS_OVERRIDE", int(train_cfg.get("max_steps", 10)))
    train_cfg["num_generations"] = env_override_int("RL_NUM_GENERATIONS", int(train_cfg.get("num_generations", 2)))
    train_cfg["per_device_train_batch_size"] = env_override_int(
        "RL_PER_DEVICE_BATCH_OVERRIDE", int(train_cfg.get("per_device_train_batch_size", train_cfg["num_generations"]))
    )
    train_cfg["gradient_accumulation_steps"] = env_override_int(
        "RL_GRAD_ACCUM_OVERRIDE", int(train_cfg.get("gradient_accumulation_steps", 1))
    )
    train_cfg["learning_rate"] = env_override_float("RL_LR_OVERRIDE", float(train_cfg.get("learning_rate", 5e-6)))
    train_cfg["vllm_gpu_memory_utilization"] = env_override_float(
        "RL_VLLM_GPU_MEMORY_UTILIZATION", float(train_cfg.get("vllm_gpu_memory_utilization", model_cfg.get("gpu_memory_utilization", 0.70)))
    )

    output_dir = env_override_str("RL_OUTPUT_DIR", train_cfg.get("output_dir"))
    if not output_dir:
        output_dir = f"/workspace/outputs/rl/{run_name}"
    train_cfg["output_dir"] = output_dir
    train_cfg.setdefault("logging_dir", f"/workspace/outputs/runs/rl/{run_name}")

    kwargs = {
        "output_dir": train_cfg.get("output_dir"),
        "logging_dir": train_cfg.get("logging_dir"),
        "learning_rate": train_cfg.get("learning_rate"),
        "per_device_train_batch_size": train_cfg.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps"),
        "max_steps": train_cfg.get("max_steps"),
        "warmup_steps": int(train_cfg.get("warmup_steps", 0)),
        "num_generations": train_cfg.get("num_generations"),
        "max_prompt_length": int(train_cfg.get("max_prompt_length", 1024)),
        "max_completion_length": int(train_cfg.get("max_completion_length", 512)),
        "logging_steps": int(train_cfg.get("logging_steps", 1)),
        "save_steps": int(train_cfg.get("save_steps", max(1, int(train_cfg.get("max_steps", 10))))),
        "save_total_limit": int(train_cfg.get("save_total_limit", 2)),
        "report_to": train_cfg.get("report_to", "tensorboard"),
        "remove_unused_columns": bool(train_cfg.get("remove_unused_columns", False)),
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "use_vllm": bool(train_cfg.get("use_vllm", True)),
        "vllm_gpu_memory_utilization": train_cfg.get("vllm_gpu_memory_utilization"),
        "beta": float(train_cfg.get("beta", 0.0)),
    }
    return GRPOConfig(**filter_kwargs(GRPOConfig, kwargs))


def build_trainer(model, tokenizer, args: GRPOConfig, train_dataset, reward_fn):
    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "reward_funcs": reward_fn,
    }
    sig = inspect.signature(GRPOTrainer)
    if "processing_class" in sig.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        kwargs["tokenizer"] = tokenizer
    return GRPOTrainer(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl/grpo_pythoncodes.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    run_name = args.run_name or cfg.get("run", {}).get("name", "grpo_pythoncodes")
    set_seed(int(cfg.get("run", {}).get("seed", 42)))

    cfg.setdefault("model", {})["local_files_only"] = bool(args.local_files_only or cfg.get("model", {}).get("local_files_only", True))

    dataset_cfg = dict(cfg.get("dataset", {}))
    dataset_cfg["limit"] = env_override_int("RL_DATASET_LIMIT_OVERRIDE", dataset_cfg.get("limit"))

    print(f"[rl] run_name={run_name}")
    print(f"[rl] base_model={cfg['model']['base_model']}")
    print(f"[rl] dataset_limit={dataset_cfg.get('limit')}")

    # DRY_RUN_STOP_BEFORE_MODEL_LOAD: dry-run validates config and dataset only.


    if getattr(args, "dry_run", False):


        print("[rl] dry-run ok: config and dataset prepared; skipping model load/trainer")


        return 0


    model, tokenizer = load_model_tokenizer(cfg)
    train_dataset = prepare_pythoncodes_grpo_dataset(dataset_cfg, tokenizer=tokenizer)
    print(f"[rl] prepared train rows={len(train_dataset)} columns={train_dataset.column_names}")

    reward_cfg = PythonRewardConfig.from_dict(cfg.get("reward", {}))
    reward_fn = make_python_reward(reward_cfg)
    training_args = build_training_args(cfg, run_name=run_name)

    print(f"[rl] GRPO output_dir={training_args.output_dir}")
    print(f"[rl] max_steps={training_args.max_steps} num_generations={getattr(training_args, 'num_generations', None)}")

    if args.dry_run:
        print("[rl] dry-run complete; not constructing trainer.train()")
        return 0

    trainer = build_trainer(model, tokenizer, training_args, train_dataset, reward_fn)
    train_result = trainer.train()

    output_adapter = Path(env_override_str("RL_ADAPTER_OUTPUT_DIR", cfg.get("model", {}).get("output_dir")) or f"/workspace/models/qwen3-4b-instruct-2507-sft-{run_name}")
    output_adapter.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_adapter))
    tokenizer.save_pretrained(str(output_adapter))

    manifest = {
        "run_name": run_name,
        "config": args.config,
        "base_model": cfg["model"]["base_model"],
        "adapter_path": str(output_adapter),
        "train_output_dir": str(training_args.output_dir),
        "train_result": getattr(train_result, "metrics", {}),
        "dataset_rows": len(train_dataset),
    }
    write_json(Path(training_args.output_dir) / "grpo_manifest.json", manifest)
    write_json(output_adapter / "grpo_manifest.json", manifest)
    print(f"[rl] saved adapter: {output_adapter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
