"""
Prompt adapter for the locally scored flytech/python-codes-25k dataset.

The scored dataset keeps the original pythoncodes fields
(instruction/input/output/text) and adds curriculum score/category columns.
Training uses the same chat structure as src/data/prompt/pythoncodes.py, but
for the current Unsloth/TRL stack we pre-tokenize and create labels manually:
system/user tokens are masked with -100, assistant tokens are trainable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

from src.config import DATASETS_DIR
from src.data.types import CodingTask

SYSTEM_PROMPT = "You are an expert Python programming assistant."
STOP_TOKENS = ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]


def _non_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_dataset(
    disk_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    split: str = "train",
) -> Dataset:
    """Load the scored pythoncodes dataset from disk or from the merged parquet."""
    if parquet_path is not None and Path(parquet_path).exists():
        return load_dataset("parquet", data_files=str(parquet_path), split=split)

    path = Path(disk_path) if disk_path is not None else DATASETS_DIR / "pythoncodes_cl_scored"
    dataset = load_from_disk(str(path))

    if isinstance(dataset, DatasetDict):
        return dataset[split] if split in dataset else dataset["train"]
    return dataset


def build_messages(example: dict, train: bool = True) -> dict[str, list[dict[str, str]]]:
    """Return the conversation used by the old pythoncodes prompt adapter."""
    user_content = _non_empty(example.get("instruction"))
    input_context = _non_empty(example.get("input"))

    if input_context:
        user_content = f"{user_content}\n\nInput:\n{input_context}" if user_content else f"Input:\n{input_context}"

    if not user_content:
        user_content = _non_empty(example.get("text"))

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if train:
        messages.append({"role": "assistant", "content": _non_empty(example.get("output"))})

    return {"messages": messages}


def build_prompt(example: dict, tokenizer: PreTrainedTokenizerBase, train: bool = False) -> dict[str, str]:
    """Compatibility helper for inference/eval code that expects rendered text."""
    messages = build_messages(example, train=train)["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not train,
    )
    return {"text": text}


def build_tokenized_chat(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    train: bool = True,
) -> dict[str, list[int]]:
    """Tokenize a chat example and mask non-assistant tokens in labels.

    This keeps assistant-only SFT semantics without relying on the broken raw
    `messages` path in the current Unsloth-patched SFTTrainer.
    """
    messages = build_messages(example, train=train)["messages"]

    if not train:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    prompt_messages = messages[:-1]
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    prompt = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = list(full["input_ids"])
    labels = list(input_ids)
    prompt_len = min(len(prompt["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": list(full["attention_mask"]),
        "labels": labels,
    }


def get_prepared_dataset(
    tokenizer: PreTrainedTokenizerBase | None = None,
    split: str = "train",
    disk_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    as_messages: bool = True,
    tokenized: bool = False,
    max_length: int = 2048,
) -> Dataset:
    dataset = get_dataset(disk_path=disk_path, parquet_path=parquet_path, split="train")

    if tokenized:
        if tokenizer is None:
            raise ValueError("tokenizer is required when tokenized=True")
        return dataset.map(
            build_tokenized_chat,
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "train": split == "train"},
            remove_columns=list(dataset.column_names),
            desc="Tokenizing scored pythoncodes chats",
        )

    if as_messages:
        return dataset.map(
            build_messages,
            fn_kwargs={"train": split == "train"},
            remove_columns=list(dataset.column_names),
            desc="Formatting scored pythoncodes conversations",
        )

    if tokenizer is None:
        raise ValueError("tokenizer is required when as_messages=False")

    return dataset.map(
        build_prompt,
        fn_kwargs={"tokenizer": tokenizer, "train": split == "train"},
        remove_columns=list(dataset.column_names),
        desc="Formatting scored pythoncodes prompts",
    )


def pythoncodes_cl_scored_to_task(row: dict, tokenizer: PreTrainedTokenizerBase) -> CodingTask:
    prompt_data = build_prompt(row, tokenizer, train=False)
    return CodingTask(
        prompt=prompt_data["text"],
        canonical_solution=row.get("output", ""),
        tests=[],
        stop_tokens=STOP_TOKENS,
    )
