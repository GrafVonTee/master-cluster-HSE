import os
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from datasets import IterableDataset, Dataset
import src.config as config
from src.logger import setup_logger


logger = setup_logger(__name__, "sgd_training.log")


def train_model_sgd(
    dataset_fn,
    assistant_only_loss=False,
    lr=1e-3,
    batch_size=2,
    gradient_steps=8,
    max_steps=500,
    warmup_steps=25
):
    logger.info(f"Загрузка модели {config.MODEL_PATH} для FULL обучения...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.MODEL_PATH,
        max_seq_length = config.MAX_TOKENS,
        dtype = None,
        load_in_4bit = False,
        full_finetuning = True,
    )
    model.gradient_checkpointing_enable()

    logger.info("Составляем датасет...")
    dataset = dataset_fn(tokenizer, split="train")

    if isinstance(dataset, IterableDataset) and dataset.column_names is None:
        required_samples = max_steps * batch_size * gradient_steps
        dataset = Dataset.from_list(list(dataset.take(required_samples)))

    run_name = "sgd_full_train_run"
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
            optim = "sgd",
            report_to = "tensorboard",
            logging_dir = tb_log_dir,
            run_name = run_name,
            dataloader_num_workers = 4,
            dataloader_pin_memory = True,
        ),
    )

    logger.info("Начинаем полное обучение (Full Fine-Tuning) с оптимизатором SGD...")
    trainer.train()

    logger.info(f"Сохраняем полные веса модели в {config.SGD_MODEL_PATH}...")
    model.save_pretrained(config.SGD_MODEL_PATH, tokenizer)
    tokenizer.save_pretrained(config.SGD_MODEL_PATH)

    logger.info("Обучение завершено!")

    return model, tokenizer
