import gc
import os
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.curriculum.cluster.io import (
    atomic_to_parquet,
    chunk_bounds,
    load_model_cached,
    load_source_dataset,
    load_tokenizer_cached,
    out_root,
    require_columns,
)


def _format_example(row: dict) -> str:
    instruction = row.get("instruction", "") or ""
    input_text = row.get("input", "") or ""
    output = row.get("output", "") or ""

    if input_text.strip():
        return (
            f"Instruction:\n{instruction}\n\n"
            f"Input:\n{input_text}\n\n"
            f"Solution:\n{output}"
        )

    return (
        f"Instruction:\n{instruction}\n\n"
        f"Solution:\n{output}"
    )


def _build_judge_prompt(row: dict, min_score: int, max_score: int) -> str:
    example = _format_example(row)

    return f"""You are evaluating the difficulty of a Python code generation training example.

Rate the example from {min_score} to {max_score}:
{min_score} = very easy
{max_score} = very hard

Consider:
- algorithmic complexity;
- amount of reasoning needed;
- code length;
- edge cases;
- required Python knowledge.

Return only one integer from {min_score} to {max_score}. Do not explain. Do not write anything except the number.

Example:
{example}

Difficulty score:"""


def _apply_chat_template_or_fallback(tokenizer, prompts: list[str]) -> list[str]:
    texts = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]

        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text = prompt
        except Exception:
            text = prompt

        texts.append(text)

    return texts


def _parse_score(text: str, min_score: int, max_score: int) -> float:
    if text is None:
        return np.nan

    text = str(text).strip()
    match = re.search(r"[-+]?\d+", text)

    if match is None:
        return np.nan

    value = int(match.group(0))
    value = max(min_score, min(max_score, value))

    return float(value)


def _batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _generate_scores(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    max_length: int,
    max_new_tokens: int,
    min_score: int,
    max_score: int,
) -> tuple[list[float], list[str]]:
    texts = _apply_chat_template_or_fallback(tokenizer, prompts)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    input_lengths = encoded["attention_mask"].sum(dim=1).tolist()

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    raw_outputs = []

    for i, output_ids in enumerate(generated):
        new_tokens = output_ids[int(input_lengths[i]):]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        raw_outputs.append(raw)

    scores = [
        _parse_score(raw, min_score=min_score, max_score=max_score)
        for raw in raw_outputs
    ]

    del encoded, generated

    return scores, raw_outputs


def score_llm_judge_chunk(cfg: dict, task_id: int | None = None) -> str | None:
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    ds = load_source_dataset(cfg)
    require_columns(ds, ["instruction", "input", "output"])

    # ВАЖНО: используем глобальный dataset.chunk_size.
    chunk_size = int(cfg["dataset"]["chunk_size"])
    start, end = chunk_bounds(len(ds), chunk_size, int(task_id))

    if start is None:
        print(f"task_id={task_id}: empty chunk, dataset len={len(ds)}", flush=True)
        return None

    out_dir = out_root(cfg) / "llm_judge_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"chunk_{int(task_id):05d}.parquet"

    if cfg.get("runtime", {}).get("skip_existing", True) and out_path.exists():
        print(f"skip existing: {out_path}", flush=True)
        return str(out_path)

    print(f"task_id={task_id}, rows={start}:{end}, n={end - start}", flush=True)

    ds = ds.select(range(start, end))

    tokenizer = load_tokenizer_cached(cfg)
    model = load_model_cached(cfg)

    device = cfg["model"].get("device", "cuda")

    judge_cfg = cfg.get("llm_judge", {})
    batch_size = int(judge_cfg.get("batch_size", 8))
    max_length = int(judge_cfg.get("max_length", 2048))
    max_new_tokens = int(judge_cfg.get("max_new_tokens", 4))
    min_score = int(judge_cfg.get("min_score", 1))
    max_score = int(judge_cfg.get("max_score", 5))

    indexed_rows = [(start + i, row) for i, row in enumerate(ds)]
    rows = []

    for batch in tqdm(
        _batched(indexed_rows, batch_size),
        total=(len(indexed_rows) + batch_size - 1) // batch_size,
        desc=f"llm_judge chunk {task_id}",
    ):
        idxs = [item[0] for item in batch]
        examples = [item[1] for item in batch]

        prompts = [
            _build_judge_prompt(row, min_score=min_score, max_score=max_score)
            for row in examples
        ]

        try:
            scores, raw_outputs = _generate_scores(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                min_score=min_score,
                max_score=max_score,
            )

        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or batch_size <= 1:
                raise

            print("OOM on judge batch. Retrying per-example.", flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            scores = []
            raw_outputs = []

            for prompt in prompts:
                one_score, one_raw = _generate_scores(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=[prompt],
                    device=device,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    min_score=min_score,
                    max_score=max_score,
                )
                scores.extend(one_score)
                raw_outputs.extend(one_raw)

        for idx, score, raw in zip(idxs, scores, raw_outputs):
            rows.append({
                "__idx": int(idx),
                "llm_judge_score": float(score) if np.isfinite(score) else np.nan,
                "llm_judge_raw": raw,
            })

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values("__idx")
    atomic_to_parquet(df, out_path)

    print(f"saved: {out_path}", flush=True)
    return str(out_path)
