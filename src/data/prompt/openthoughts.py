import src.config as config
from datasets import load_dataset, Features, Value
from src.data.types import CodingTask
import os
import re
import glob
from huggingface_hub import snapshot_download


def get_dataset(streaming=True):
    return load_dataset(
        "open-thoughts/OpenThoughts-114k",
        split="train",
        streaming=streaming,
        cache_dir=config.DATASETS_DIR,
    )


def get_prepared_dataset(tokenizer, split="train"):
    dataset = get_dataset()
    dataset = dataset.shuffle(seed=42, buffer_size=10000)

    val_size = 1000

    if split != "train":
        dataset = dataset.take(val_size)
    else:
        dataset = dataset.skip(val_size)

    dataset = dataset.map(
        build_prompt,
        fn_kwargs={"tokenizer": tokenizer, "train": (split == "train")},
        remove_columns=["system", "conversations"],
        features=Features({"text": Value("string")}),
    )

    def compute_length(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=False,
            padding=False,
            return_length=True
        )
        return {"input_len": tokenized["length"]}

    dataset = dataset.map(compute_length, batched=True)
    max_len = config.MAX_TOKENS
    dataset = dataset.filter(lambda x: x["input_len"] <= max_len)
    dataset.remove_columns(["input_len"])

    return dataset


def build_prompt(example: dict, tokenizer, train=False) -> str:
    messages = []

    raw_messages = example.get("conversations", [])
    system_prompt = example.get("system", "")

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for msg in raw_messages:
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        role = msg.get("from") or msg.get("role")
        content = msg.get("value") or msg.get("content")

        if content:
            content = re.sub(r"<\|begin_of_thought\|>", "<think>", content)
            content = re.sub(r"<\|end_of_thought\|>", "</think>", content)
            content = re.sub(r"<\|begin_of_solution\|>", "", content)
            content = re.sub(r"<\|end_of_solution\|>", "", content)

        normalized_role = role_map.get(role, role)
        messages.append({"role": normalized_role, "content": content})

    if not train:
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not train
    )

    return {"text": text}


def openthoughts_to_task(row: dict, tokenizer) -> CodingTask:
    prompt_data = build_prompt(row, tokenizer, train=False)

    canonical_solution = ""
    raw_msgs = row.get("conversations", [])
    for msg in reversed(raw_msgs):
        role = msg.get("from") or msg.get("role")
        if role in ["gpt", "assistant"]:
            canonical_solution = msg.get("value") or msg.get("content")
            break

    return CodingTask(
        prompt=prompt_data['text'],
        canonical_solution=canonical_solution,
        tests=[], # нету ;-;
        stop_tokens=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"],
    )
