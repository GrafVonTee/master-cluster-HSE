import gc
import math
import os
import traceback
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .base import BaseScorer
from src.data.prompt.pythoncodes import build_prompt
from src.data.curriculum.cluster.io import (
    atomic_to_parquet,
    chunk_bounds,
    load_model_cached,
    load_source_dataset,
    load_tokenizer_cached,
    out_root,
    require_columns,
)


class PlainPPLScorer(BaseScorer):
    """Plain LM loss / PPL for the full train text.

    This is intentionally different from the old conditional target-only PPL:
        old: loss(answer | prompt)
        new: loss(full chat training text)

    For curriculum sorting, use ppl_loss. ppl_score = exp(ppl_loss) is stored
    only for readability; exp is monotonic, so it gives the same ordering as
    ppl_loss but is numerically less convenient.
    """

    def __init__(
        self,
        name: str,
        tokenizer: Any,
        model: Any,
        build_prompt_fn: Any,
        device: str = "cuda",
        max_length: int = 2048,
        skip_truncated: bool = True,
    ):
        super().__init__(name)
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.device = device
        self.build_prompt_fn = build_prompt_fn
        self.max_length = int(max_length)
        self.skip_truncated = bool(skip_truncated)

    def _get_plain_loss(self, text: str) -> float:
        ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        if len(ids) < 2:
            return float("nan")

        if len(ids) > self.max_length:
            if self.skip_truncated:
                return float("nan")
            ids = ids[: self.max_length]

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        labels = input_ids.clone()

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = float(outputs.loss.item())

        del input_ids, labels, outputs
        return loss

    def score(self, examples: dict) -> list[float]:
        scores: list[float] = []
        batch_size = len(examples["instruction"])

        for i in range(batch_size):
            row = {k: v[i] for k, v in examples.items()}

            try:
                full_data = self.build_prompt_fn(row, self.tokenizer, train=True)
                loss = self._get_plain_loss(full_data["text"])
                scores.append(float(loss))
            except Exception:
                print(f"\n[PlainPPL Error] example={i}", flush=True)
                traceback.print_exc()
                scores.append(float("nan"))

        return scores


def _safe_exp(loss: float) -> float:
    if not np.isfinite(loss):
        return np.nan
    return float(math.exp(min(float(loss), 50.0)))


def _batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _batch_plain_lm_loss(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    max_length: int,
    skip_truncated: bool = True,
) -> tuple[list[float], list[int], list[bool]]:
    """Return per-example full-text LM loss.

    Loss is averaged only over non-padding tokens. Examples longer than
    max_length are skipped by default instead of silently scoring truncated
    text, because truncated outputs make PPL categories noisy.
    """
    encoded_items: list[tuple[int, list[int], bool]] = []
    losses = [float("nan")] * len(texts)
    token_counts = [0] * len(texts)
    was_truncated = [False] * len(texts)

    for i, text in enumerate(texts):
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        token_counts[i] = len(ids)

        if len(ids) < 2:
            continue

        if len(ids) > max_length:
            was_truncated[i] = True
            if skip_truncated:
                continue
            ids = ids[:max_length]

        encoded_items.append((i, ids, was_truncated[i]))

    if not encoded_items:
        return losses, token_counts, was_truncated

    max_len = max(len(ids) for _, ids, _ in encoded_items)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    batch_ids = []
    attention = []

    for _, ids, _ in encoded_items:
        pad_len = max_len - len(ids)
        batch_ids.append(ids + [pad_id] * pad_len)
        attention.append([1] * len(ids) + [0] * pad_len)

    input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention, dtype=torch.long, device=device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().bool()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        denom = shift_mask.sum(dim=1).clamp(min=1)
        row_losses = (token_losses * shift_mask).sum(dim=1) / denom

    values = row_losses.detach().float().cpu().numpy().tolist()

    for (original_idx, _, _), loss in zip(encoded_items, values):
        losses[original_idx] = float(loss)

    del input_ids, attention_mask, outputs, logits
    del shift_logits, shift_labels, shift_mask, token_losses, row_losses

    return losses, token_counts, was_truncated


def score_plain_ppl_chunk(cfg: dict, task_id: int | None = None) -> str | None:
    """Score one dataset chunk with plain full-text PPL.

    Output intentionally keeps the old directory name `ppl_ifd_chunks` so the
    existing Slurm jobs and merge script do not need to be renamed. The IFD
    columns are kept as NaN compatibility placeholders.
    """
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

    print(f"plain PPL task_id={task_id}, rows={start}:{end}, n={end - start}", flush=True)

    ds = ds.select(range(start, end))

    tokenizer = load_tokenizer_cached(cfg)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model_cached(cfg)

    device = cfg["model"].get("device", "cuda")
    batch_size = int(cfg.get("ppl", {}).get("batch_size", cfg.get("loss", {}).get("batch_size", 4)))
    max_length = int(cfg.get("ppl", {}).get("max_length", cfg.get("loss", {}).get("max_length", 2048)))
    skip_truncated = bool(cfg.get("ppl", {}).get("skip_truncated", True))

    indexed_rows = [(start + i, row) for i, row in enumerate(ds)]
    rows = []

    for batch in tqdm(list(_batched(indexed_rows, batch_size)), desc=f"plain ppl chunk {task_id}"):
        idxs = [item[0] for item in batch]
        examples = [item[1] for item in batch]
        full_texts = [build_prompt(row, tokenizer, train=True)["text"] for row in examples]

        try:
            ppl_losses, token_counts, truncated_flags = _batch_plain_lm_loss(
                model=model,
                tokenizer=tokenizer,
                texts=full_texts,
                device=device,
                max_length=max_length,
                skip_truncated=skip_truncated,
            )
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or batch_size <= 1:
                raise

            print("OOM on plain PPL batch. Retrying per-example.", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            ppl_losses = []
            token_counts = []
            truncated_flags = []

            for text in full_texts:
                loss, n_tok, was_trunc = _batch_plain_lm_loss(
                    model=model,
                    tokenizer=tokenizer,
                    texts=[text],
                    device=device,
                    max_length=max_length,
                    skip_truncated=skip_truncated,
                )
                ppl_losses.extend(loss)
                token_counts.extend(n_tok)
                truncated_flags.extend(was_trunc)

        for idx, ppl_loss, token_count, was_truncated in zip(idxs, ppl_losses, token_counts, truncated_flags):
            rows.append(
                {
                    "__idx": int(idx),
                    "ppl_loss": float(ppl_loss) if np.isfinite(ppl_loss) else np.nan,
                    "ppl_score": _safe_exp(float(ppl_loss)) if np.isfinite(ppl_loss) else np.nan,
                    "ppl_num_tokens": int(token_count),
                    "ppl_truncated": bool(was_truncated),
                    # Compatibility placeholders. IFD is not used for the new PPL split.
                    "ifd_score": np.nan,
                    "cond_loss": np.nan,
                    "uncond_loss": np.nan,
                }
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    atomic_to_parquet(df, out_path)

    print(f"saved: {out_path}", flush=True)
    return str(out_path)
