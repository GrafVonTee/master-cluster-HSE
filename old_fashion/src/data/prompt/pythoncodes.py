from src.config import DATASETS_DIR
from datasets import load_dataset, Features, Value
from src.data.types import CodingTask

def get_dataset(streaming=True):
    return load_dataset(
        "flytech/python-codes-25k",
        split="train",
        streaming=streaming,
        cache_dir=DATASETS_DIR,
    )

def get_prepared_dataset(tokenizer, split="train"):
    dataset = get_dataset()
    dataset = dataset.shuffle(seed=42, buffer_size=10000)
    val_size = 1000

    if split != "train":
        dataset = dataset.take(val_size)
    else:
        dataset = dataset.skip(val_size)

    return dataset.map(
        build_prompt,
        fn_kwargs={"tokenizer": tokenizer, "train": (split == "train")},
        remove_columns=["instruction", "input", "output", "text"],
        features=Features({"text": Value("string")}),
    )

def build_prompt(example: dict, tokenizer, train=False) -> str:
    user_content = example.get("instruction", "")
    input_context = example.get("input", "")

    if input_context and input_context.strip():
        user_content += f"\n\nInput:\n{input_context}"

    messages = [
        {"role": "system", "content": "You are an expert Python programming assistant."},
        {"role": "user", "content": user_content}
    ]

    if train:
        assistant_content = example.get("output", "")
        messages.append({"role": "assistant", "content": assistant_content})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not train,
        # thinking=True,
    )

    return {"text": text}

def flytech_to_task(row: dict, tokenizer) -> CodingTask:
    prompt_data = build_prompt(row, tokenizer, train=False)
    canonical_solution = row.get("output", "")

    return CodingTask(
        prompt=prompt_data['text'],
        canonical_solution=canonical_solution,
        tests=[],
        stop_tokens=["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"],
    )
