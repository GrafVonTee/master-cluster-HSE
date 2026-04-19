import os
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from datasets import IterableDataset, Dataset
import src.config as config
from src.logger import setup_logger


logger = setup_logger(__name__, "training.log")


def train_model_pipelinel(training_stages):
    """
    Выполняет последовательное обучение.

    Ожидаемый формат training_stages:
    [
        ("Dataset_1_Name", dataset_fn_1, {
            "lr": 2e-5, "pretrain": True, "load_in_4bit": True, "dtype": None,
            "train_batch_size": 4, "gradient_steps": 4, "max_steps": 1000, "warmup_steps": 50
        }),
        ("Dataset_2_Name", dataset_fn_2, {
            "lr": 1e-5, "pretrain": False, "load_in_4bit": True, "dtype": None,
            "train_batch_size": 4, "gradient_steps": 4, "max_steps": 500, "warmup_steps": 25
        })
    ]
    """
    model = None
    tokenizer = None

    for stage_idx, (dataset_name, dataset_fn, params) in enumerate(training_stages):
        logger.info(f"=== Запуск этапа {stage_idx + 1}/{len(training_stages)}: {dataset_name} ===")

        if model is None:
            logger.info(f"Загрузка модели {config.MODEL_PATH}...")
            load_in_4bit = params.get("load_in_4bit", True)
            dtype = params.get("dtype", None)

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = config.MODEL_PATH,
                max_seq_length = config.MAX_TOKENS,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
            )

            model = FastLanguageModel.get_peft_model(
                model,
                r = 32,
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
                lora_alpha = 32,
                lora_dropout = 0,
                bias = "none",
                use_gradient_checkpointing = "unsloth",
            )
        else:
            logger.info("Модель уже загружена, продолжаем обучение (веса сохраняются)...")

        logger.info(f"Составляем датасет для {dataset_name}...")
        dataset = dataset_fn(tokenizer, split="train")

        if isinstance(dataset, IterableDataset) and dataset.column_names is None:
            required_samples = params["max_steps"] * params["train_batch_size"] * params["gradient_steps"]
            logger.info(f"Берем {required_samples} примеров из IterableDataset.")
            dataset = Dataset.from_list(list(dataset.take(required_samples)))

        run_name = f"stage_{stage_idx + 1}_{dataset_name}"
        tb_log_dir = os.path.join("runs", run_name)

        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = config.MAX_TOKENS,
            packing = False,
            assistant_only_loss = params["assistant_only_loss"],
            args = TrainingArguments(
                per_device_train_batch_size = params["train_batch_size"],
                gradient_accumulation_steps = params["gradient_steps"],
                max_steps = params["max_steps"],
                warmup_steps = params["warmup_steps"],
                learning_rate = params["lr"],
                lr_scheduler_type="cosine",
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 10,
                output_dir = f"checkpoints/{run_name}",
                optim = "adamw_8bit",
                report_to = "tensorboard",
                logging_dir = tb_log_dir,
                run_name = run_name,
                dataloader_num_workers = 4,
                dataloader_pin_memory = True,
            ),
        )

        logger.info(f"Начинаем обучение на {dataset_name}...")
        trainer.train()

    logger.info(f"Сохраняем объединенную модель в {config.SFT_MODEL_PATH}...")
    model.save_pretrained(config.SFT_MODEL_PATH, tokenizer)
    tokenizer.save_pretrained(config.SFT_MODEL_PATH)

    logger.info("Все этапы обучения завершены!")

    return model, tokenizer


def train_model(
        dataset_fn,
        lr=1e-5,
        load_in_4bit=True,
        train_batch_size=4,
        accumulation_steps=4,
        max_steps=500,
        assistant_only_loss = False,
    ):
    logger.info(f"Загрузка модели {config.MODEL_PATH}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.MODEL_PATH,
        max_seq_length = config.MAX_TOKENS,
        dtype = None,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    logger.info("Составляем датасет...")
    dataset = dataset_fn(tokenizer, split="train")

    if isinstance(dataset, IterableDataset) and dataset.column_names is None:
        dataset = Dataset.from_list(list(dataset.take(max_steps * train_batch_size * accumulation_steps)))

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = config.MAX_TOKENS,
        packing = False,
        assistant_only_loss = assistant_only_loss,
        args = TrainingArguments(
            per_device_train_batch_size = train_batch_size,
            gradient_accumulation_steps = accumulation_steps,
            max_steps = max_steps,
            warmup_steps = 25,
            learning_rate = lr,
            lr_scheduler_type="cosine",
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 25,
            output_dir = "checkpoints",
            optim = "adamw_8bit",
            report_to = "none", # Отключаем wandb
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        ),
    )

    logger.info("Начинаем обучение...")
    trainer.train()

    logger.info(f"Сохраняем объединенную модель в {config.SFT_MODEL_PATH}...")
    model.save_pretrained(config.SFT_MODEL_PATH, tokenizer)
    tokenizer.save_pretrained(config.SFT_MODEL_PATH)

    logger.info("Обучение завершено!")

    return model, tokenizer
