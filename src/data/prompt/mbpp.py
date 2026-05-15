from datasets import load_dataset

from src.config import DATASETS_DIR
from src.data.types import CodingTask


def get_dataset():
    return load_dataset("google-research-datasets/mbpp", "sanitized", cache_dir=DATASETS_DIR)


def get_prepared_dataset(tokenizer, split="train"):
    mbpp = get_dataset()
    return mbpp[split].map(
        build_prompt,
        fn_kwargs={"tokenizer": tokenizer, "train": (split == "train")},
    )


def extract_signature_from_mbpp_code(code: str) -> str:
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("def "):
            return line.rstrip(":")
    raise ValueError("No function signature found")


def _apply_chat_template(tokenizer, messages, train: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": not train,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_prompt(example: dict, tokenizer, train=False) -> dict:
    task_text = example["prompt"]
    code_solution = example["code"]
    signature_line = extract_signature_from_mbpp_code(example["code"])

    system_msg = (
        "You are an expert Python coding assistant. "
        "Given a problem description and function signature, implement the function so that it passes all tests."
    )
    user_msg = (
        "Problem:\n"
        f"{task_text}\n\n"
        "Use the following function signature:\n"
        f"{signature_line}:\n\n"
        "Write the full Python function implementation. "
        "Do not change the function name or arguments. "
        "Return only the final solution wrapped in a markdown Python code block."
    )
    assistant_msg = f"```python\n{code_solution}\n```"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    if train:
        messages.append({"role": "assistant", "content": assistant_msg})

    text = _apply_chat_template(tokenizer, messages, train=train)
    return {"text": text}


def mbpp_to_task(row: dict, tokenizer) -> CodingTask:
    prompt_data = build_prompt(row, tokenizer, train=False)
    return CodingTask(
        prompt=prompt_data["text"],
        canonical_solution=row["code"],
        tests=row["test_list"],
        stop_tokens=["\nclass", "\ndef", "\nif", "\nprint"],
    )
