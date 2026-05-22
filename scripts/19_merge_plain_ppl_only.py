from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from datasets import load_from_disk, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_DIR / p


def make_categories(values: pd.Series) -> pd.Series:
    valid = values.replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid) == 0:
        return pd.Series(["unknown"] * len(values), index=values.index)

    q1 = valid.quantile(1 / 3)
    q2 = valid.quantile(2 / 3)

    def cat(x):
        if pd.isna(x) or np.isinf(x):
            return "unknown"
        if x <= q1:
            return "easy"
        if x <= q2:
            return "medium"
        return "hard"

    return values.map(cat)


def main():
    config_path = PROJECT_DIR / "configs/score_pythoncodes_cluster.yaml"

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_path = resolve_path(
        cfg.get("dataset", {}).get("local_path", "datasets/pythoncodes_cl_scored")
    )

    chunk_dir = resolve_path(
        cfg.get("outputs", {}).get(
            "ppl_ifd_chunk_dir",
            "outputs/curriculum_array/ppl_ifd_chunks",
        )
    )

    out_path = dataset_path

    print(f"Loading base dataset: {dataset_path}")
    ds = load_from_disk(str(dataset_path))
    base_df = pd.DataFrame(ds)
    base_df["__idx"] = np.arange(len(base_df), dtype=int)

    chunk_files = sorted(chunk_dir.glob("*.parquet"))
    if not chunk_files:
        raise RuntimeError(f"No PPL chunk parquet files found in {chunk_dir}")

    print(f"Found {len(chunk_files)} PPL chunk files:")
    for p in chunk_files:
        print(f"  {p}")

    parts = []
    for p in chunk_files:
        parts.append(pq.read_table(p).to_pandas())

    ppl_df = pd.concat(parts, ignore_index=True)

    if "__idx" not in ppl_df.columns:
        raise RuntimeError("PPL chunks must contain __idx column")

    keep_cols = [
        "__idx",
        "ppl_loss",
        "ppl_score",
        "ppl_num_tokens",
        "ppl_truncated",
        "model_entropy",
        "model_confidence",
    ]
    keep_cols = [c for c in keep_cols if c in ppl_df.columns]

    ppl_df = ppl_df[keep_cols].drop_duplicates("__idx", keep="last")

    print(f"Base rows: {len(base_df)}")
    print(f"PPL rows:  {len(ppl_df)}")

    old_ppl_cols = [
        "ppl_loss",
        "ppl_score",
        "ppl_category",
        "ppl_num_tokens",
        "ppl_truncated",
        "model_entropy",
        "model_confidence",
    ]

    base_df = base_df.drop(
        columns=[c for c in old_ppl_cols if c in base_df.columns],
        errors="ignore",
    )

    merged = base_df.merge(ppl_df, on="__idx", how="left", validate="one_to_one")

    if "ppl_loss" not in merged.columns:
        raise RuntimeError("Merged dataset has no ppl_loss column")

    merged["ppl_category"] = make_categories(merged["ppl_loss"])

    missing = merged["ppl_loss"].isna().sum()
    print(f"Missing ppl_loss rows: {missing}")

    print("ppl_category counts:")
    print(merged["ppl_category"].value_counts(dropna=False))

    print("ppl_loss describe:")
    print(merged["ppl_loss"].describe())

    merged = merged.drop(columns=["__idx"])

    tmp_path = out_path.parent / (out_path.name + ".tmp_plain_ppl_merge")
    if tmp_path.exists():
        import shutil
        shutil.rmtree(tmp_path)

    print(f"Saving temp dataset: {tmp_path}")
    Dataset.from_pandas(merged, preserve_index=False).save_to_disk(str(tmp_path))

    backup_path = out_path.parent / (out_path.name + ".before_plain_ppl_merge")
    if backup_path.exists():
        import shutil
        shutil.rmtree(backup_path)

    print(f"Moving old dataset to backup: {backup_path}")
    out_path.rename(backup_path)

    print(f"Moving temp dataset to final: {out_path}")
    tmp_path.rename(out_path)

    print("Done.")


if __name__ == "__main__":
    main()
