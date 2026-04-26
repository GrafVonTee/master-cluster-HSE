import gc
import os

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

    return f"""You are a strict difficulty classifier for Python code generation training examples.

Choose exactly one label:
{min_score} = very easy
2 = easy
3 = medium
4 = hard
{max_score} = very hard

Return only one digit: {min_score}, 2, 3, 4, or {max_score}.

Example:
{example}

Difficulty label:"""


def _apply_chat_template_or_fallback(tokenizer, prompts: list[str]) -> list[str]:
    texts = []

    for prompt in prompts:
        # /no_think помогает Qwen3 не уходить в reasoning-текст.
        prompt = prompt + "\n/no_think"

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


def _batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _candidate_token_ids(tokenizer, min_score: int, max_score: int) -> dict[int, list[int]]:
    out = {}

    for score in range(min_score, max_score + 1):
        variants = [
            str(score),
            f" {score}",
            f"\n{score}",
            f"\n\n{score}",
        ]

        ids = []

        for text in variants:
            token_ids = tokenizer.encode(text, add_special_tokens=False)

            if len(token_ids) >= 1:
                ids.append(int(token_ids[-1]))

        out[score] = sorted(set(ids))

    return out


def _forced_choice_scores(
    model,
    tokenizer,
    prompts: list[str],
    device: str,
    max_length: int,
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

    attention_mask = encoded["attention_mask"]
    last_positions = attention_mask.sum(dim=1) - 1

    candidate_ids = _candidate_token_ids(tokenizer, min_score, max_score)

    with torch.inference_mode():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=attention_mask,
            use_cache=False,
        )

        logits = outputs.logits
        next_logits = logits[
            torch.arange(logits.shape[0], device=logits.device),
            last_positions,
            :,
        ].float()

        batch_scores = []
        raw_outputs = []

        for row_logits in next_logits:
            score_to_logit = {}

            for score, ids in candidate_ids.items():
                if not ids:
                    score_to_logit[score] = -float("inf")
                    continue

                token_logits = row_logits[torch.tensor(ids, device=row_logits.device)]
                score_to_logit[score] = float(token_logits.max().detach().cpu().item())

            best_score = max(score_to_logit, key=score_to_logit.get)
            batch_scores.append(float(best_score))
            raw_outputs.append(f"forced_choice:{best_score}; logits={score_to_logit}")

    del encoded, attention_mask, outputs, logits, next_logits

    return batch_scores, raw_outputs


def score_llm_judge_chunk(cfg: dict, task_id: int | None = None) -> str | None:
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    ds = load_source_dataset(cfg)
    require_columns(ds, ["instruction", "input", "output"])

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
    model.config.use_cache = False

    device = cfg["model"].get("device", "cuda")

    judge_cfg = cfg.get("llm_judge", {})
    batch_size = int(judge_cfg.get("batch_size", 8))
    max_length = int(judge_cfg.get("max_length", 2048))
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
            scores, raw_outputs = _forced_choice_scores(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                max_length=max_length,
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
                one_score, one_raw = _forced_choice_scores(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=[prompt],
                    device=device,
                    max_length=max_length,
                    min_score=min_score,
                    max_score=max_score,
                )
                scores.extend(one_score)
                raw_outputs.extend(one_raw)

        for idx, score, raw in zip(idxs, scores, raw_outputs):
            rows.append({
                "__idx": int(idx),
                "llm_judge_score": float(score),
                "llm_judge_raw": raw,
            })

        gc.collect()

    df = pd.DataFrame(rows).sort_values("__idx")
    atomic_to_parquet(df, out_path)

    print(f"saved: {out_path}", flush=True)
    return str(out_path)
