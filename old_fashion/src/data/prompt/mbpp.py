from src.config import DATASETS_DIR
from datasets import load_dataset
from src.data.types import CodingTask


def get_dataset():
    return load_dataset("google-research-datasets/mbpp", "sanitized", cache_dir=DATASETS_DIR)


def get_prepared_dataset(tokenizer, split="train"):
    mbpp = get_dataset()
    return mbpp[split].map(
        build_prompt,
        fn_kwargs={"tokenizer": tokenizer, "train": (split == "train")}
    )


def extract_signature_from_mbpp_code(code: str) -> str:
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("def "):
            return line.rstrip(":")
    raise ValueError("No function signature found")


def build_prompt(example: dict, tokenizer, train=False) -> str:
    task_text = example["prompt"]
    code_solution = example["code"]
    signature_line = extract_signature_from_mbpp_code(example["code"])

    system_msg = (
        "You are an expert Python coding assistant. "
        "Given a problem description and function signature, "
        "implement the function body so that it passes all tests."
    )
    user_msg = (
        "Problem:\n"
        f"{task_text}\n\n"
        "Use the following function signature:\n"
        f"{signature_line}:\n\n"
        "Write the full Python function implementation. "
        "Do NOT change the function name or arguments. "
        "You must analyze the problem in a <think> block first, then provide the solution. "
        "The code must be wrapped in markdown block."
    )
    assistant_msg = (
        f"```python\n{code_solution}\n```"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    if train:
        messages.append({"role": "assistant", "content": assistant_msg})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not train,
        thinking=True
    )

    return {"text": text}


def mbpp_to_task(row: dict, tokenizer) -> CodingTask:
    """Преобразует строку MBPP в CodingTask"""
    prompt_data = build_prompt(row, tokenizer, train=False)

    return CodingTask(
        prompt=prompt_data['text'],
        canonical_solution=row['code'],
        tests=row['test_list'], # В MBPP это уже список строк
        stop_tokens=["\nclass", "\ndef", "\nif", "\nprint"]
    )
