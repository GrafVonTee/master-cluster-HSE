from src.config import DATASETS_DIR
from datasets import load_dataset
from src.data.types import CodingTask
import re
from typing import List, Optional


def get_dataset():
    return load_dataset("openai/openai_humaneval", cache_dir=DATASETS_DIR)


def get_prepared_dataset(tokenizer):
    humaneval = get_dataset()
    return humaneval.map(build_prompt, fn_kwargs={"tokenizer": tokenizer})


def build_prompt(example: dict, tokenizer) -> str:
    task_id = example.get("task_id", "")
    task_text = example["prompt"].rstrip()
    entry_point = example.get("entry_point", "")

    system_msg = (
        "You are an expert Python coding assistant. "
        "You will be given a Python file snippet containing imports and a single function "
        "signature with a docstring. Complete the function implementation so it is correct."
    )

    user_msg = (
        f"Task ID: {task_id}\n"
        f"Function to implement: {entry_point}\n\n"
        "Complete the following code by writing the function body.\n"
        "1. First, analyze the problem and plan your solution inside <think>...</think> tags.\n"
        "2. Then, provide the implementation inside a markdown code block (```python ... ```).\n"
        "\nConstraints:\n"
        "- Keep all existing imports, the function name, and its arguments unchanged.\n"
        "- Do not modify the docstring.\n"
        "- You may add local helper functions if needed, but do not change the target signature.\n\n"
        "Code:\n"
        f"{task_text}\n"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def extract_tests(test_str: str) -> List[str]:
    """Парсит asserts из строки тестов (для HumanEval)."""
    if not test_str:
        return []

    lines = test_str.splitlines()
    tests: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Ваша логика с балансом скобок для многострочных assert
        if re.match(r"^\s*assert\b", line):
            buf = [line.rstrip()]
            i += 1
            text = "\n".join(buf)
            balance = text.count("(") - text.count(")")
            balance += text.count("[") - text.count("]")
            balance += text.count("{") - text.count("}")

            while i < len(lines) and balance > 0:
                buf.append(lines[i].rstrip())
                text = "\n".join(buf)
                balance = text.count("(") - text.count(")")
                balance += text.count("[") - text.count("]")
                balance += text.count("{") - text.count("}")
                i += 1
            tests.append("\n".join(buf).strip())
        else:
            i += 1
    return tests


def humaneval_to_task(row: dict, tokenizer) -> CodingTask:
    prompt_str = build_prompt(row, tokenizer)

    # ВАЖНО: Используем парсер здесь.
    # Теперь executor получит список ['assert x==1', 'assert x==2']
    # Добавляем контекст candidate = entry_point, который был в test_execute
    entry_point = row['entry_point']
    raw_tests = extract_tests(row['test'])
    prepared_tests = [f"candidate = {entry_point}\n{t}" for t in raw_tests]

    return CodingTask(
        prompt=prompt_str,
        canonical_solution=row['canonical_solution'],
        tests=prepared_tests,
        stop_tokens=["\nclass", "\ndef", "\n#", "if __name__"]
    )
