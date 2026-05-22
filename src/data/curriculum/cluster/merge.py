import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset

from src.data.curriculum.cluster.io import (
    PROJECT_DIR,
    atomic_write_json,
    load_source_dataset,
    out_root,
)


def assign_categories(scores, percentiles: list[float]) -> tuple[list[str], list[float | None]]:
    arr = np.asarray(scores, dtype=float)
    valid = arr[np.isfinite(arr)]

    if len(valid) == 0:
        return ["medium"] * len(arr), [None, None]

    p_low, p_high = np.percentile(valid, percentiles)

    categories = []

    for x in arr:
        if not np.isfinite(x):
            categories.append("medium")
        elif x <= p_low:
            categories.append("easy")
        elif x <= p_high:
            categories.append("medium")
        else:
            categories.append("hard")

    return categories, [float(p_low), float(p_high)]


def _read_chunked_parquets(chunks_dir: Path, required: bool = True) -> pd.DataFrame | None:
    files = sorted(chunks_dir.glob("chunk_*.parquet"))

    if not files:
        if required:
            raise RuntimeError(f"No parquet chunks found in {chunks_dir}")
        return None

    return pd.concat(
        [pd.read_parquet(path) for path in files],
        ignore_index=True,
    )


def merge_curriculum_scores(cfg: dict) -> dict:
    root = out_root(cfg)

    ds = load_source_dataset(cfg)
    base_df = pd.DataFrame(ds)

    old_score_cols = [
        "length_score",
        "ppl_score",
        "ppl_loss",
        "ifd_score",
        "cond_loss",
        "uncond_loss",
        "lexical_cluster_score",
        "semantic_cluster_score",
        "llm_judge_score",
        "llm_judge_raw",
    ]

    old_category_cols = [
        col for col in base_df.columns
        if col.endswith("_category")
    ]

    drop_cols = [
        col for col in old_score_cols + old_category_cols
        if col in base_df.columns
    ]

    if drop_cols:
        base_df = base_df.drop(columns=drop_cols)

    base_df["__idx"] = np.arange(len(base_df), dtype=int)

    length_path = root / "length.parquet"
    lexical_path = root / "lexical.parquet"
    semantic_path = root / "semantic.parquet"
    ppl_ifd_chunks_dir = root / "ppl_ifd_chunks"
    llm_judge_chunks_dir = root / "llm_judge_chunks"

    required_files = [
        length_path,
        lexical_path,
        semantic_path,
        ppl_ifd_chunks_dir,
        llm_judge_chunks_dir,
    ]

    for path in required_files:
        if not path.exists():
            raise RuntimeError(f"Missing required score file: {path}")

    parts = [
        pd.read_parquet(length_path),
        pd.read_parquet(lexical_path),
        pd.read_parquet(semantic_path),
        _read_chunked_parquets(ppl_ifd_chunks_dir, required=True),
        _read_chunked_parquets(llm_judge_chunks_dir, required=True),
    ]

    df = base_df

    for part in parts:
        if "__idx" not in part.columns:
            raise ValueError(f"Missing __idx column in score part: columns={part.columns.tolist()}")

        part = part.drop_duplicates("__idx")
        df = df.merge(part, on="__idx", how="left")

    score_columns = [
        "length_score",
        "ppl_loss",
        "ppl_score",
        "ifd_score",
        "lexical_cluster_score",
        "semantic_cluster_score",
        "llm_judge_score",
    ]

    percentiles = cfg["curriculum"].get("percentiles", [33, 66])

    thresholds = {}

    for col in score_columns:
        if col not in df.columns:
            continue

        # For PPL curriculum we sort by loss. PPL=exp(loss) has the same order,
        # but loss is numerically safer and easier to debug.
        if col == "ppl_score" and "ppl_loss" in df.columns:
            continue

        categories, threshold = assign_categories(df[col].to_numpy(), percentiles)

        if col == "ppl_loss":
            base_name = "ppl"
        else:
            base_name = col.replace("_score", "")

        df[f"{base_name}_category"] = categories
        thresholds[col] = threshold

    final_dataset_dir = PROJECT_DIR / cfg["output"]["final_dataset_dir"]
    final_parquet_path = PROJECT_DIR / cfg["output"]["final_parquet_path"]
    final_preview_csv_path = PROJECT_DIR / cfg["output"]["final_preview_csv_path"]
    final_summary_path = PROJECT_DIR / cfg["output"]["final_summary_path"]

    final_dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    final_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    final_preview_csv_path.parent.mkdir(parents=True, exist_ok=True)
    final_summary_path.parent.mkdir(parents=True, exist_ok=True)

    if final_dataset_dir.exists():
        shutil.rmtree(final_dataset_dir)

    output_df = df.drop(columns=["__idx"])

    final_ds = Dataset.from_pandas(output_df, preserve_index=False)
    final_ds.save_to_disk(str(final_dataset_dir))

    output_df.to_parquet(final_parquet_path, index=False)
    output_df.head(200).to_csv(final_preview_csv_path, index=False)

    category_counts = {
        col: {str(k): int(v) for k, v in output_df[col].value_counts(dropna=False).to_dict().items()}
        for col in output_df.columns
        if col.endswith("_category")
    }

    missing_scores = {
        col: int(output_df[col].isna().sum())
        for col in score_columns
        if col in output_df.columns
    }

    summary = {
        "num_rows": int(len(output_df)),
        "score_columns": score_columns,
        "thresholds": thresholds,
        "category_counts": category_counts,
        "missing_scores": missing_scores,
        "outputs": {
            "dataset_dir": str(final_dataset_dir),
            "parquet": str(final_parquet_path),
            "preview_csv": str(final_preview_csv_path),
            "summary": str(final_summary_path),
        },
    }

    atomic_write_json(final_summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    return summary
