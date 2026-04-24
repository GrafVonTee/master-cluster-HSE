import gc
import os
from pathlib import Path

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

from src.data.prompt.pythoncodes import build_prompt


def _batch_target_loss(
    model,
    tokenizer,
    full_texts: list[str],
    prompt_texts: list[str],
    device: str,
    max_length: int,
) -> list[float]:
    encoded = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    prompt_lens = []
    for prompt in prompt_texts:
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )["input_ids"]
        prompt_lens.append(len(prompt_ids))

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    for i, prompt_len in enumerate(prompt_lens):
        real_len = int(attention_mask[i].sum().item())

        if prompt_len >= real_len:
            prompt_len = max(0, real_len - 1)

        labels[i, :prompt_len] = -100

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        mask = shift_labels.ne(-100)
        denom = mask.sum(dim=1).clamp(min=1)
        losses = (token_losses * mask).sum(dim=1) / denom

    result = losses.detach().float().cpu().numpy().tolist()

    del encoded, input_ids, attention_mask, labels
    del outputs, logits, shift_logits, shift_labels
    del token_losses, mask, losses

    return result


def _batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _safe_exp(x: float) -> float:
    if not np.isfinite(x):
        return np.nan

    # Защита от overflow. PPL как score всё равно остаётся "очень большой".
    if x > 80:
        return float(np.exp(80))

    return float(np.exp(x))


def score_ppl_ifd_chunk(cfg: dict, task_id: int | None = None) -> str | None:
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    ds = load_source_dataset(cfg)
    require_columns(ds, ["instruction", "input", "output"])

    chunk_size = int(cfg["dataset"]["chunk_size"])
    start, end = chunk_bounds(len(ds), chunk_size, int(task_id))

    if start is None:
        print(f"task_id={task_id}: empty chunk, dataset len={len(ds)}", flush=True)
        return None

    out_dir = out_root(cfg) / "ppl_ifd_chunks"
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
    batch_size = int(cfg["loss"].get("batch_size", 4))
    max_length = int(cfg["loss"].get("max_length", 2048))

    indexed_rows = [(start + i, row) for i, row in enumerate(ds)]
    rows = []

    for batch in tqdm(list(_batched(indexed_rows, batch_size)), desc=f"ppl_ifd chunk {task_id}"):
        idxs = [item[0] for item in batch]
        examples = [item[1] for item in batch]

        prompt_texts = [build_prompt(row, tokenizer, train=False)["text"] for row in examples]
        full_texts = [build_prompt(row, tokenizer, train=True)["text"] for row in examples]
        output_texts = [(row.get("output", "") or "") for row in examples]
        empty_prompts = [""] * len(examples)

        try:
            cond_losses = _batch_target_loss(
                model=model,
                tokenizer=tokenizer,
                full_texts=full_texts,
                prompt_texts=prompt_texts,
                device=device,
                max_length=max_length,
            )

            uncond_losses = _batch_target_loss(
                model=model,
                tokenizer=tokenizer,
                full_texts=output_texts,
                prompt_texts=empty_prompts,
                device=device,
                max_length=max_length,
            )

        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or batch_size <= 1:
                raise

            print("OOM on batch. Retrying per-example.", flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            cond_losses = []
            uncond_losses = []

            for row in examples:
                prompt_text = build_prompt(row, tokenizer, train=False)["text"]
                full_text = build_prompt(row, tokenizer, train=True)["text"]
                output_text = row.get("output", "") or ""

                cond_losses.extend(
                    _batch_target_loss(
                        model=model,
                        tokenizer=tokenizer,
                        full_texts=[full_text],
                        prompt_texts=[prompt_text],
                        device=device,
                        max_length=max_length,
                    )
                )

                uncond_losses.extend(
                    _batch_target_loss(
                        model=model,
                        tokenizer=tokenizer,
                        full_texts=[output_text],
                        prompt_texts=[""],
                        device=device,
                        max_length=max_length,
                    )
                )

        for idx, cond_loss, uncond_loss in zip(idxs, cond_losses, uncond_losses):
            ppl_score = _safe_exp(float(cond_loss))

            if not np.isfinite(uncond_loss) or float(uncond_loss) == 0.0:
                ifd_score = np.nan
            else:
                ifd_score = float(cond_loss) / float(uncond_loss)

            rows.append({
                "__idx": int(idx),
                "ppl_score": float(ppl_score),
                "ifd_score": float(ifd_score) if np.isfinite(ifd_score) else np.nan,
                "cond_loss": float(cond_loss),
                "uncond_loss": float(uncond_loss),
            })

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    atomic_to_parquet(df, out_path)

    print(f"saved: {out_path}", flush=True)
    return str(out_path)
