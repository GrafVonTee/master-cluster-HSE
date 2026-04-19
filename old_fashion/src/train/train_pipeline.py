import os
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from datasets import IterableDataset, Dataset
import src.config as config
from src.logger import setup_logger

logger = setup_logger(__name__, "pipeline_training.log")

def train_pipeline(training_stages, use_lora=True, save_path=None):
    """
    Универсальная функция для обучения (как последовательного, так и одиночного).
    Поддерживает Full Fine-Tuning (SGD/AdamW) и LoRA.

    Ожидаемый формат training_stages:
    [
        ("Имя_Датасета", dataset_fn, {
            "lr": 1e-4, "batch_size": 2, "gradient_steps": 8,
            "max_steps": 500, "warmup_steps": 25,
            "optim": "adamw_8bit", "assistant_only_loss": True
        })
    ]
    """
    model = None
    tokenizer = None

    for stage_idx, (dataset_name, dataset_fn, params) in enumerate(training_stages):
        logger.info(f"=== Запуск этапа {stage_idx + 1}/{len(training_stages)}: {dataset_name} ===")

        batch_size = params.get("batch_size", params.get("train_batch_size", 4))
        gradient_steps = params.get("gradient_steps", params.get("accumulation_steps", 4))
        max_steps = params.get("max_steps", 500)
        warmup_steps = params.get("warmup_steps", 25)
        lr = params.get("lr", 1e-4 if use_lora else 1e-5)
        optim = params.get("optim", "adamw" if use_lora else "sgd")
        assistant_only_loss = params.get("assistant_only_loss", False)

        if model is None:
            logger.info(f"Загрузка базовой модели {config.MODEL_PATH}...")

            # Если Full FT, отключаем 4-bit квантизацию принудительно
            load_in_4bit = params.get("load_in_4bit", True) if use_lora else False
            dtype = params.get("dtype", None)

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = config.MODEL_PATH,
                max_seq_length = config.MAX_TOKENS,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
                full_finetuning = not use_lora, # Флаг для Full FT
            )

            if use_lora:
                logger.info("Инициализация PEFT/LoRA адаптеров...")
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
                logger.info("Настройка модели для Full Fine-Tuning...")
                model.gradient_checkpointing_enable()
        else:
            logger.info("Модель уже загружена, продолжаем пайплайн...")

        logger.info(f"Составляем датасет для {dataset_name}...")
        dataset = dataset_fn(tokenizer, split="train")

        if isinstance(dataset, IterableDataset) and dataset.column_names is None:
            required_samples = max_steps * batch_size * gradient_steps
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
            assistant_only_loss = assistant_only_loss,
            args = TrainingArguments(
                per_device_train_batch_size = batch_size,
                gradient_accumulation_steps = gradient_steps,
                max_steps = max_steps,
                warmup_steps = warmup_steps,
                learning_rate = lr,
                lr_scheduler_type = "cosine",
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 10,
                output_dir = f"checkpoints/{run_name}",
                optim = optim,
                report_to = "tensorboard",
                logging_dir = tb_log_dir,
                run_name = run_name,
                dataloader_num_workers = 4,
                dataloader_pin_memory = True,
            ),
        )

        logger.info(f"Начинаем обучение на {dataset_name} (Оптимизатор: {optim})...")
        trainer.train()

    if save_path is None:
        save_path = config.SFT_MODEL_PATH if use_lora else config.SGD_MODEL_PATH

    logger.info(f"Сохраняем объединенную модель в {save_path}...")
    model.save_pretrained(save_path, tokenizer)
    tokenizer.save_pretrained(save_path)

    logger.info("Все этапы пайплайна успешно завершены!")

    return model, tokenizer
