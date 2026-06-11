"""
Prompt adapter for synthetic Clingo/ASP code-generation tasks.

Expected fields:
  instruction, facts, output/reference, difficulty, topic

The training target is assistant-only Clingo code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

from src.config import DATASETS_DIR
from src.data.types import CodingTask

SYSTEM_PROMPT = (
    "You are an expert Answer Set Programming assistant. "
    "Return only valid clingo/ASP code. "
    "Do not explain. Do not use markdown."
)

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
    path = Path(disk_path) if disk_path is not None else DATASETS_DIR / "clingo" / "synthetic_v2_lora_train"
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        return dataset[split] if split in dataset else dataset["train"]
    return dataset


def build_messages(example: dict, train: bool = True) -> dict[str, list[dict[str, str]]]:
    instruction = _non_empty(example.get("instruction"))
    facts = _non_empty(example.get("facts"))

    user_content = f"Task:\n{instruction}\n\nFacts:\n{facts}\n\nWrite the clingo program."

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if train:
        target = _non_empty(example.get("reference")) or _non_empty(example.get("output"))
        messages.append({"role": "assistant", "content": target})

    return {"messages": messages}


def _apply_chat_template(tokenizer, messages, *, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def build_prompt(example: dict, tokenizer, train: bool = False) -> dict[str, str]:
    messages = build_messages(example, train=train)["messages"]
    return {"text": _apply_chat_template(tokenizer, messages, add_generation_prompt=not train)}


def build_tokenized_chat(
    example: dict,
    tokenizer,
    max_length: int,
    train: bool = True,
) -> dict[str, list[int]]:
    if not train:
        prompt = build_prompt(example, tokenizer=tokenizer, train=False)["text"]
        encoded = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    user_messages = build_messages(example, train=False)["messages"]
    full_messages = build_messages(example, train=True)["messages"]

    prompt_text = _apply_chat_template(tokenizer, user_messages, add_generation_prompt=True)
    full_text = _apply_chat_template(tokenizer, full_messages, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = list(input_ids)

    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    if all(x == -100 for x in labels) and labels:
        keep = min(64, len(labels))
        labels[-keep:] = input_ids[-keep:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def get_prepared_dataset(
    tokenizer=None,
    split: str = "train",
    disk_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    as_messages: bool = True,
) -> Dataset:
    dataset = get_dataset(disk_path=disk_path, parquet_path=parquet_path, split=split)

    if as_messages:
        return dataset.map(
            build_messages,
            fn_kwargs={"train": split == "train"},
            remove_columns=list(dataset.column_names),
            desc="Formatting clingo conversations",
        )

    if tokenizer is None:
        raise ValueError("tokenizer is required when as_messages=False")

    return dataset.map(
        build_prompt,
        fn_kwargs={"tokenizer": tokenizer, "train": split == "train"},
        remove_columns=list(dataset.column_names),
        desc="Formatting clingo prompts",
    )


def clingo_synthetic_to_task(row: dict, tokenizer) -> CodingTask:
    prompt_data = build_prompt(row, tokenizer, train=False)
    return CodingTask(
        prompt=prompt_data["text"],
        canonical_solution=row.get("reference", row.get("output", "")),
        tests=[],
        stop_tokens=STOP_TOKENS,
    )
