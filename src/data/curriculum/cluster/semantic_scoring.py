import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
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


def _save_npz_atomic(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)

    tmp.replace(path)


def _get_embeddings(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    max_length: int,
    layer_idx: int,
) -> np.ndarray:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[layer_idx]
        attention_mask = inputs["attention_mask"]

        last_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)

        vectors = hidden[batch_idx, last_idx]
        embeddings = vectors.detach().float().cpu().numpy()

    del inputs, outputs, hidden, attention_mask, vectors

    return embeddings


def embed_semantic_chunk(cfg: dict, task_id: int | None = None) -> str | None:
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    ds = load_source_dataset(cfg)
    require_columns(ds, ["output"])

    chunk_size = int(cfg["dataset"]["chunk_size"])
    start, end = chunk_bounds(len(ds), chunk_size, int(task_id))

    if start is None:
        print(f"task_id={task_id}: empty chunk, dataset len={len(ds)}", flush=True)
        return None

    out_dir = out_root(cfg) / "semantic_emb_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"chunk_{int(task_id):05d}.npz"

    if cfg.get("runtime", {}).get("skip_existing", True) and out_path.exists():
        print(f"skip existing: {out_path}", flush=True)
        return str(out_path)

    print(f"task_id={task_id}, rows={start}:{end}, n={end - start}", flush=True)

    ds = ds.select(range(start, end))

    tokenizer = load_tokenizer_cached(cfg)
    model = load_model_cached(cfg)

    model.config.output_hidden_states = True

    device = cfg["model"].get("device", "cuda")
    max_length = int(cfg["semantic"].get("max_length", 1024))
    batch_size = int(cfg["semantic"].get("batch_size", 8))

    layer_idx = cfg["semantic"].get("layer_idx")
    if layer_idx is None:
        layer_idx = model.config.num_hidden_layers // 2
    else:
        layer_idx = int(layer_idx)

    outputs = ds["output"]

    all_idx = []
    all_emb = []

    for local_start in tqdm(range(0, len(outputs), batch_size), desc=f"semantic chunk {task_id}"):
        local_end = min(local_start + batch_size, len(outputs))
        texts = outputs[local_start:local_end]

        emb = _get_embeddings(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            max_length=max_length,
            layer_idx=layer_idx,
        )

        all_emb.append(emb)
        all_idx.extend(range(start + local_start, start + local_end))

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    idx = np.array(all_idx, dtype=np.int64)
    emb = np.vstack(all_emb).astype("float32")

    _save_npz_atomic(out_path, idx=idx, emb=emb)

    print(f"saved: {out_path}, emb_shape={emb.shape}", flush=True)
    return str(out_path)


def score_semantic_from_embeddings(cfg: dict) -> str:
    root = out_root(cfg)
    emb_dir = root / "semantic_emb_chunks"
    out_path = root / "semantic.parquet"

    if cfg.get("runtime", {}).get("skip_existing", True) and out_path.exists():
        print(f"skip existing: {out_path}", flush=True)
        return str(out_path)

    files = sorted(emb_dir.glob("chunk_*.npz"))

    if not files:
        raise RuntimeError(f"No semantic embedding chunks found in {emb_dir}")

    idxs = []
    embs = []

    for path in files:
        data = np.load(path)
        idxs.append(data["idx"])
        embs.append(data["emb"])

    idx = np.concatenate(idxs)
    emb = np.vstack(embs).astype("float32")

    order = np.argsort(idx)
    idx = idx[order]
    emb = emb[order]

    n_clusters = int(cfg["semantic"].get("n_clusters", 10))

    print(f"KMeans on semantic embeddings: emb={emb.shape}, n_clusters={n_clusters}", flush=True)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    kmeans.fit(emb)

    distances = kmeans.transform(emb)
    scores = np.min(distances, axis=1)

    df = pd.DataFrame({
        "__idx": idx.astype(int),
        "semantic_cluster_score": scores.astype(float),
    })

    atomic_to_parquet(df, out_path)

    print(f"saved: {out_path}", flush=True)
    return str(out_path)
