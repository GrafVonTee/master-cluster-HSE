import functools
import gc
import inspect
import json
import os
from pathlib import Path

import datasets
import pandas as pd
import torch
import yaml
from datasets import Dataset, Features, IterableDataset, Value, load_from_disk
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

import src.config as config
from src.data.loader import load_benchmark
from src.data.prompt import humaneval, mbpp, pythoncodes
from src.evaluator import Evaluator
from src.inference.vllm_inference import setup_model
from src.logger import setup_logger
from src.metrics import GreedyPass, MeanEntropy, PassAtk, PercentPassed


logger = setup_logger(__name__, "training.log")


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_train_config(path="configs/train/lora_pythoncodes_cl.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _training_args_eval_key():
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        return "eval_strategy"
    return "evaluation_strategy"


def _grad_norm(model):
    total = 0.0
    has_grad = False

    for p in model.parameters():
        if p.grad is None:
            continue

        has_grad = True
        total += p.grad.detach().data.norm(2).item() ** 2

    return total ** 0.5 if has_grad else None


class LoggedSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs, *args, **kwargs):
        loss = super().training_step(model, inputs, *args, **kwargs)

        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            norm = _grad_norm(model)
            if norm is not None:
                self.log({"grad_norm_manual": norm})

        return loss


def _make_lora_model(train_cfg):
    model_cfg = train_cfg["model"]
    lora_cfg = train_cfg["lora"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_PATH,
        max_seq_length=config.MAX_TOKENS,
        dtype=model_cfg.get("dtype"),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.get("r", 32),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0),
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
    )

    return model, tokenizer


def _to_sft_dataset(df, tokenizer):
    dataset = Dataset.from_pandas(df, preserve_index=False)

    return dataset.map(
        pythoncodes.build_prompt,
        fn_kwargs={"tokenizer": tokenizer, "train": True},
        remove_columns=dataset.column_names,
        features=Features({"text": Value("string")}),
    )


def load_pythoncodes_cl_df(train_cfg=None):
    train_cfg = train_cfg or load_train_config()
    dataset_cfg = train_cfg["dataset"]
    dataset_name = dataset_cfg.get("name", "pythoncodes_cl_scored")

    dataset_path = Path(config.DATASETS_DIR) / dataset_name
    parquet_path = Path(dataset_cfg.get("parquet_path", ""))
    if not parquet_path.is_absolute():
        parquet_path = Path(config.PROJECT_DIR) / parquet_path

    if dataset_path.exists():
        df = load_from_disk(str(dataset_path)).to_pandas()
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        raise FileNotFoundError(f"Не найден CL-датасет: {dataset_path} или {parquet_path}")

    limit = dataset_cfg.get("limit")
    if limit:
        df = df.head(int(limit))

    return df.reset_index(drop=True)


def _split_train_val(df, train_cfg):
    dataset_cfg = train_cfg["dataset"]
    val_size = int(dataset_cfg.get("val_size", 1000))
    seed = int(dataset_cfg.get("seed", 42))

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    val_df = df.iloc[:val_size].reset_index(drop=True)
    train_df = df.iloc[val_size:].reset_index(drop=True)

    return train_df, val_df


def _stage_dfs(train_df, run_cfg, train_cfg):
    category_col = run_cfg.get("category_col")

    if category_col is None:
        return [(run_cfg["name"], train_df)]

    if category_col not in train_df.columns:
        raise ValueError(f"Нет колонки {category_col}. Доступные колонки: {list(train_df.columns)}")

    stages = run_cfg.get("stages") or train_cfg["curriculum"].get("stages", ["easy", "medium", "hard"])
    existing = set(train_df[category_col].dropna().unique())

    if not set(stages).issubset(existing):
        stages = sorted(existing)

    result = []
    prev = []
    cumulative = run_cfg.get("cumulative", train_cfg["curriculum"].get("cumulative", False))

    for stage in stages:
        part = train_df[train_df[category_col] == stage].reset_index(drop=True)

        if cumulative:
            prev.append(part)
            part = pd.concat(prev, ignore_index=True)

        if len(part):
            result.append((str(stage), part))

    return result


def _make_training_args(train_cfg, run_cfg, stage_name, run_dir):
    t = train_cfg["training"]

    max_steps = run_cfg.get("max_steps", t.get("max_steps", 600))
    if run_cfg.get("category_col") is not None:
        max_steps = run_cfg.get("stage_max_steps", t.get("stage_max_steps", max_steps))

    kwargs = {
        "output_dir": str(run_dir / "checkpoints" / stage_name),
        "per_device_train_batch_size": t.get("train_batch_size", 1),
        "gradient_accumulation_steps": t.get("gradient_steps", 8),
        "max_steps": max_steps,
        "warmup_steps": t.get("warmup_steps", 25),
        "learning_rate": t.get("lr", 2e-4),
        "lr_scheduler_type": t.get("lr_scheduler_type", "cosine"),
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "logging_steps": t.get("logging_steps", 10),
        "save_steps": t.get("save_steps", 100),
        "save_total_limit": t.get("save_total_limit", 2),
        "optim": t.get("optim", "adamw_8bit"),
        "report_to": t.get("report_to", "tensorboard"),
        "logging_dir": str(run_dir / "tb" / stage_name),
        "run_name": f"{run_cfg['name']}_{stage_name}",
        "dataloader_num_workers": t.get("dataloader_num_workers", 2),
        "dataloader_pin_memory": t.get("dataloader_pin_memory", True),
        "max_grad_norm": t.get("max_grad_norm", 1.0),
        "seed": train_cfg["dataset"].get("seed", 42),
    }

    kwargs[_training_args_eval_key()] = "steps"
    kwargs["eval_steps"] = t.get("eval_steps", 50)

    return TrainingArguments(**kwargs)


def train_lora_experiment(run_cfg, train_cfg=None, df=None):
    train_cfg = train_cfg or load_train_config()
    df = df if df is not None else load_pythoncodes_cl_df(train_cfg)

    train_df, val_df = _split_train_val(df, train_cfg)
    run_name = run_cfg["name"]

    run_dir = Path(config.OUTPUTS_DIR) / "train_runs" / run_name
    adapter_dir = Path(config.MODELS_DIR) / f"{config.SELECTED_MODEL}-{run_name}"

    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== {run_name} ===")
    logger.info(f"model={config.MODEL_PATH}")
    logger.info(f"adapter={adapter_dir}")
    logger.info(f"train={len(train_df)}, val={len(val_df)}")

    model, tokenizer = _make_lora_model(train_cfg)
    val_ds = _to_sft_dataset(val_df, tokenizer)

    logs = []
    stages = _stage_dfs(train_df, run_cfg, train_cfg)

    for stage_name, stage_df in stages:
        logger.info(f"stage={stage_name}; rows={len(stage_df)}")
        train_ds = _to_sft_dataset(stage_df, tokenizer)

        trainer = LoggedSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            dataset_text_field="text",
            max_seq_length=config.MAX_TOKENS,
            packing=train_cfg["training"].get("packing", False),
            assistant_only_loss=train_cfg["training"].get("assistant_only_loss", True),
            args=_make_training_args(train_cfg, run_cfg, stage_name, run_dir),
        )

        trainer.train()
        trainer.evaluate()

        log_df = pd.DataFrame(trainer.state.log_history)
        log_df["experiment"] = run_name
        log_df["stage"] = stage_name
        logs.append(log_df)

        del trainer, train_ds
        cleanup_cuda()

    metrics = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    metrics.to_parquet(run_dir / "metrics.parquet", index=False)

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    summary = {
        "experiment": run_name,
        "model": config.MODEL_PATH,
        "adapter": str(adapter_dir),
        "metrics_csv": str(run_dir / "metrics.csv"),
        "metrics_parquet": str(run_dir / "metrics.parquet"),
        "stages": [{"name": name, "rows": len(stage_df)} for name, stage_df in stages],
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    del model, tokenizer, val_ds
    cleanup_cuda()

    return summary


def train_lora_experiments(train_cfg=None, runs=None, only=None, skip=0):
    train_cfg = train_cfg or load_train_config()
    runs = runs or train_cfg["runs"]
    df = load_pythoncodes_cl_df(train_cfg)

    summaries = []

    for i, run_cfg in enumerate(runs):
        if i < skip:
            continue

        if only is not None and run_cfg["name"] not in only:
            continue

        print(f"=== Experiment {i + 1}/{len(runs)}: {run_cfg['name']} ===")
        summaries.append(train_lora_experiment(run_cfg, train_cfg=train_cfg, df=df))

    return pd.DataFrame(summaries)


def collect_training_table(runs_root=None):
    runs_root = Path(runs_root or Path(config.OUTPUTS_DIR) / "train_runs")
    rows = []

    for summary_path in sorted(runs_root.glob("*/summary.json")):
        with open(summary_path, encoding="utf-8") as f:
            row = json.load(f)

        metrics = pd.read_csv(row["metrics_csv"])
        last = metrics.dropna(how="all").tail(1).to_dict("records")
        rows.append({**row, **(last[0] if last else {})})

    df = pd.DataFrame(rows)

    if len(df):
        df.to_csv(runs_root / "summary.csv", index=False)
        df.to_parquet(runs_root / "summary.parquet", index=False)

    return df


def prepare_lora_experiments(include_base=True, train_cfg=None):
    train_cfg = train_cfg or load_train_config()
    rows = []

    if include_base:
        rows.append((config.SELECTED_MODEL, config.MODEL_PATH, None))

    for run_cfg in train_cfg["runs"]:
        adapter = Path(config.MODELS_DIR) / f"{config.SELECTED_MODEL}-{run_cfg['name']}"
        if adapter.exists():
            rows.append((run_cfg["name"], config.MODEL_PATH, str(adapter)))

    return rows


def cleanup_vllm():
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass

    cleanup_cuda()


def evaluate_lora_experiments(experiments, max_tasks=None, out_name="cl_eval"):
    raw_mbpp = datasets.load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    raw_he = datasets.load_dataset("openai/openai_humaneval", split="test")

    if max_tasks:
        raw_mbpp = raw_mbpp.select(range(min(max_tasks, len(raw_mbpp))))
        raw_he = raw_he.select(range(min(max_tasks, len(raw_he))))

    metrics = [
        GreedyPass(),
        PassAtk(k=1, n_samples=1),
        PercentPassed(),
        MeanEntropy(),
    ]

    records = []

    for i, (exp_name, model_path, adapter_path) in enumerate(experiments):
        print(f"=== Eval {i + 1}/{len(experiments)}: {exp_name} ===")

        llm, tokenizer, _ = setup_model(model_path=model_path, adapter_path=adapter_path)
        evaluator = Evaluator(llm, tokenizer, metrics)

        for benchmark, raw, mapper_fn in [
            ("mbpp", raw_mbpp, mbpp.mbpp_to_task),
            ("humaneval", raw_he, humaneval.humaneval_to_task),
        ]:
            mapper = functools.partial(mapper_fn, tokenizer=tokenizer)
            tasks = load_benchmark(raw, mapper)
            result = evaluator.run(tasks)

            records.append({
                "experiment": exp_name,
                "benchmark": benchmark,
                "model_path": model_path,
                "adapter_path": adapter_path,
                **result,
            })

        del llm, tokenizer, evaluator
        cleanup_vllm()

    out_dir = Path(config.OUTPUTS_DIR) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / f"{out_name}.csv", index=False)
    df.to_parquet(out_dir / f"{out_name}.parquet", index=False)

    return df


def train_model_pipelinel(training_stages):
    """
    Старый интерфейс оставлен для старых ноутбуков.
    """
    model = None
    tokenizer = None

    for stage_idx, (dataset_name, dataset_fn, params) in enumerate(training_stages):
        logger.info(f"=== Запуск этапа {stage_idx + 1}/{len(training_stages)}: {dataset_name} ===")

        if model is None:
            logger.info(f"Загрузка модели {config.MODEL_PATH}...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.MODEL_PATH,
                max_seq_length=config.MAX_TOKENS,
                dtype=params.get("dtype", None),
                load_in_4bit=params.get("load_in_4bit", True),
            )

            model = FastLanguageModel.get_peft_model(
                model,
                r=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=32,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )

        dataset = dataset_fn(tokenizer, split="train")

        if isinstance(dataset, IterableDataset) and dataset.column_names is None:
            need = params["max_steps"] * params["train_batch_size"] * params["gradient_steps"]
            dataset = Dataset.from_list(list(dataset.take(need)))

        run_name = f"stage_{stage_idx + 1}_{dataset_name}"

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=config.MAX_TOKENS,
            packing=False,
            assistant_only_loss=params["assistant_only_loss"],
            args=TrainingArguments(
                per_device_train_batch_size=params["train_batch_size"],
                gradient_accumulation_steps=params["gradient_steps"],
                max_steps=params["max_steps"],
                warmup_steps=params["warmup_steps"],
                learning_rate=params["lr"],
                lr_scheduler_type="cosine",
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10,
                output_dir=f"checkpoints/{run_name}",
                optim="adamw_8bit",
                report_to="tensorboard",
                logging_dir=os.path.join("runs", run_name),
                run_name=run_name,
                dataloader_num_workers=4,
                dataloader_pin_memory=True,
            ),
        )

        trainer.train()

    model.save_pretrained(config.SFT_MODEL_PATH)
    tokenizer.save_pretrained(config.SFT_MODEL_PATH)
    return model, tokenizer


def train_model(
    dataset_fn,
    lr=1e-5,
    load_in_4bit=True,
    train_batch_size=4,
    accumulation_steps=4,
    max_steps=500,
    assistant_only_loss=False,
):
    """
    Старый интерфейс оставлен для старых ноутбуков.
    """
    logger.info(f"Загрузка модели {config.MODEL_PATH}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_PATH,
        max_seq_length=config.MAX_TOKENS,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    dataset = dataset_fn(tokenizer, split="train")

    if isinstance(dataset, IterableDataset) and dataset.column_names is None:
        dataset = Dataset.from_list(list(dataset.take(max_steps * train_batch_size * accumulation_steps)))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_TOKENS,
        packing=False,
        assistant_only_loss=assistant_only_loss,
        args=TrainingArguments(
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=accumulation_steps,
            max_steps=max_steps,
            warmup_steps=25,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=25,
            output_dir="checkpoints",
            optim="adamw_8bit",
            report_to="none",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        ),
    )

    trainer.train()
    model.save_pretrained(config.SFT_MODEL_PATH)
    tokenizer.save_pretrained(config.SFT_MODEL_PATH)

    return model, tokenizer
