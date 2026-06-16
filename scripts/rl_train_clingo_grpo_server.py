#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
from typing import Any

# Must be set before importing Unsloth.
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from src.rl.grpo_utils import read_yaml, set_seed
from src.rl.clingo_dataset import prepare_clingo_grpo_dataset
from src.dsl.clingo.rewards import ClingoRewardConfig, score_clingo_completion


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def resolve_workspace_path(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: resolve_workspace_path(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_workspace_path(v) for v in value]
    if not isinstance(value, str):
        return value
    if value == "/workspace":
        return str(PROJECT_ROOT)
    if value.startswith("/workspace/"):
        return str(PROJECT_ROOT / value.removeprefix("/workspace/"))
    return value


def filter_dataclass_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in dataclasses.fields(cls)}
    return {k: v for k, v in kwargs.items() if k in names}


def completion_to_text(completion: Any) -> str:
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def make_reward_func(cfg: ClingoRewardConfig):
    def reward_func(completions, task=None, **kwargs):
        n = len(completions)
        tasks = task if isinstance(task, list) else [task] * n

        scores = []
        for completion, task_row in zip(completions, tasks):
            text = completion_to_text(completion)
            if task_row is None:
                task_row = {}
            scores.append(score_clingo_completion(text, task_row, cfg))
        return scores

    return reward_func


def force_trainable_lora_graph(model):
    """Make loaded PEFT adapters usable for RL/SFT backward.

    Fresh LoRA goes through FastLanguageModel.get_peft_model(...), which prepares
    the training graph. Existing adapters loaded via PeftModel.from_pretrained(...)
    need the same train-mode/input-grad setup explicitly.
    """
    try:
        FastLanguageModel.for_training(model)
        print("[grpo-server] FastLanguageModel.for_training applied")
    except Exception as e:
        print(f"[grpo-server] FastLanguageModel.for_training skipped: {e!r}")

    model.train()

    try:
        model.config.use_cache = False
    except Exception:
        pass

    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
            print("[grpo-server] enable_input_require_grads applied")
        except Exception as e:
            print(f"[grpo-server] enable_input_require_grads failed: {e!r}")
    else:
        try:
            emb = model.get_input_embeddings()

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            emb.register_forward_hook(make_inputs_require_grad)
            print("[grpo-server] input embedding grad hook registered")
        except Exception as e:
            print(f"[grpo-server] input embedding grad hook failed: {e!r}")

    # Be explicit for PEFT adapters. This is harmless for fresh LoRA and important
    # for adapters loaded from disk.
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if (
            "lora_" in name
            or ".lora_A." in name
            or ".lora_B." in name
            or "modules_to_save" in name
        ):
            param.requires_grad_(True)
        if param.requires_grad:
            trainable += param.numel()

    print(f"[grpo-server] force_trainable_lora_graph: trainable={trainable:,} total={total:,}")
    if trainable == 0:
        raise RuntimeError("No trainable parameters after loading/preparing LoRA adapter")

    return model


def load_model_tokenizer(
    base_model: str,
    *,
    local_files_only: bool,
    init_adapter: str | None,
    lora_cfg: dict[str, Any],
):
    import torch

    dtype = torch.float16

    print(f"[grpo-server] loading train model: {base_model}")
    print("[grpo-server] fast_inference=False; vLLM is external server, not colocated")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=int(lora_cfg.get("max_seq_length", 2048)),
        dtype=dtype,
        load_in_4bit=bool(lora_cfg.get("load_in_4bit", False)),
        fast_inference=False,
        local_files_only=local_files_only,
    )

    if init_adapter:
        from peft import PeftModel

        print(f"[grpo-server] loading initial LoRA adapter as trainable: {init_adapter}")
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        print("[grpo-server] creating fresh LoRA adapter")
        peft_kwargs = dict(
            r=int(lora_cfg.get("r", 32)),
            target_modules=lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.0)),
            bias=str(lora_cfg.get("bias", "none")),
            use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
            random_state=int(lora_cfg.get("random_state", 3407)),
        )
        model = FastLanguageModel.get_peft_model(model, **peft_kwargs)

    model = force_trainable_lora_graph(model)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl/grpo_clingo_optimization.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--init-adapter", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = resolve_workspace_path(read_yaml(args.config))

    run_name = (
        args.run_name
        or os.environ.get("GRPO_RUN_NAME")
        or cfg.get("run_name")
        or "grpo_clingo_server"
    )

    model_cfg = dict(cfg.get("model", {}))
    dataset_cfg = dict(cfg.get("dataset", {}))
    lora_cfg = dict(cfg.get("lora", {}))
    train_cfg = dict(cfg.get("training", {}))

    base_model = (
        args.base_model
        or os.environ.get("GRPO_BASE_MODEL")
        or os.environ.get("RL_BASE_MODEL")
        or model_cfg.get("base_model")
    )
    if not base_model:
        raise ValueError("No base model provided. Set model.base_model or GRPO_BASE_MODEL.")

    init_adapter = (
        args.init_adapter
        or os.environ.get("RL_INIT_ADAPTER")
        or os.environ.get("GRPO_INIT_ADAPTER")
        or model_cfg.get("init_adapter")
    )

    dataset_limit = env_int("RL_DATASET_LIMIT_OVERRIDE", int(dataset_cfg.get("limit", 256)))
    dataset_cfg["limit"] = dataset_limit

    max_steps = env_int("RL_MAX_STEPS_OVERRIDE", int(train_cfg.get("max_steps", 10)))

    output_root = Path(str(resolve_workspace_path(cfg.get("output_root", "/workspace/outputs/rl"))))
    output_dir = output_root / run_name

    base_slug = Path(str(base_model)).name.rstrip("/")
    adapter_output_dir = Path(
        os.environ.get(
            "GRPO_ADAPTER_OUTPUT_DIR",
            str(resolve_workspace_path(f"/workspace/models/{base_slug}-sft-{run_name}")),
        )
    )

    print(f"[grpo-server] run_name={run_name}")
    print(f"[grpo-server] base_model={base_model}")
    print(f"[grpo-server] init_adapter={init_adapter}")
    print(f"[grpo-server] dataset_limit={dataset_limit}")
    print(f"[grpo-server] output_dir={output_dir}")
    print(f"[grpo-server] adapter_output_dir={adapter_output_dir}")

    set_seed(int(train_cfg.get("seed", 3407)))

    if args.dry_run:
        ds = prepare_clingo_grpo_dataset(dataset_cfg, tokenizer=None)
        print(f"[grpo-server] dry-run ok: rows={len(ds)} columns={ds.column_names}")
        return 0

    model, tokenizer = load_model_tokenizer(
        str(base_model),
        local_files_only=args.local_files_only,
        init_adapter=str(init_adapter) if init_adapter else None,
        lora_cfg=lora_cfg,
    )

    ds = prepare_clingo_grpo_dataset(dataset_cfg, tokenizer=tokenizer)
    if len(ds) == 0:
        raise RuntimeError("Prepared GRPO dataset is empty.")

    reward_cfg = ClingoRewardConfig(**{k: v for k, v in dict(cfg.get("reward", {})).items() if k in {"timeout", "max_models", "error_reward", "min_chars", "max_chars"}})

    training_kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        max_steps=max_steps,
        learning_rate=float(train_cfg.get("learning_rate", 5e-6)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "linear")),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 2)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        num_generations=int(train_cfg.get("num_generations", 2)),
        max_prompt_length=int(train_cfg.get("max_prompt_length", 1024)),
        max_completion_length=int(train_cfg.get("max_completion_length", 512)),
        beta=float(train_cfg.get("beta", 0.0)),
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        save_steps=int(train_cfg.get("save_steps", max(1, max_steps))),
        save_strategy=str(train_cfg.get("save_strategy", "no" if max_steps <= 1 else "steps")),
        report_to=train_cfg.get("report_to", []),
        remove_unused_columns=False,
        fp16=True,
        bf16=False,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=os.environ.get("GRPO_VLLM_SERVER_HOST", "127.0.0.1"),
        vllm_server_port=env_int("GRPO_VLLM_SERVER_PORT", 8000),
        vllm_group_port=env_int("GRPO_VLLM_GROUP_PORT", 51216),
        vllm_server_timeout=env_float("GRPO_VLLM_SERVER_TIMEOUT", 300.0),
    )

    training_kwargs = filter_dataclass_kwargs(GRPOConfig, training_kwargs)

    print(f"[grpo-server] GRPOConfig kwargs={training_kwargs}")

    training_args = GRPOConfig(**training_kwargs)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        reward_funcs=[make_reward_func(reward_cfg)],
        processing_class=tokenizer,
    )

    trainer.train()

    adapter_output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[grpo-server] saving adapter to {adapter_output_dir}")
    trainer.model.save_pretrained(str(adapter_output_dir))
    tokenizer.save_pretrained(str(adapter_output_dir))

    print("[grpo-server] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
