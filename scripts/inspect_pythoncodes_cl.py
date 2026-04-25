from pathlib import Path

import pandas as pd
from datasets import load_from_disk


ROOT = Path.home() / "projects/mouse-learning/master-cluster-HSE"

dataset_path = ROOT / "datasets/pythoncodes_cl_scored"
parquet_path = ROOT / "outputs/curriculum_array/pythoncodes_cl_scored.parquet"
summary_path = ROOT / "outputs/curriculum_array/pythoncodes_cl_summary.json"


print("===== PATHS =====")
print("dataset_path:", dataset_path, dataset_path.exists())
print("parquet_path:", parquet_path, parquet_path.exists())
print("summary_path:", summary_path, summary_path.exists())


print("\n===== LOAD HF DATASET =====")
ds = load_from_disk(str(dataset_path))

print(ds)
print("num rows:", len(ds))
print("columns:")
for c in ds.column_names:
    print(" ", c)


print("\n===== FIRST ROW =====")
row = ds[0]
for k, v in row.items():
    if isinstance(v, str) and len(v) > 500:
        print(f"{k}: {v[:500]!r} ...")
    else:
        print(f"{k}: {v!r}")


print("\n===== LOAD PARQUET HEAD =====")
df = pd.read_parquet(parquet_path)

print(df.shape)
print(df.head(3).T)


print("\n===== SCORE / CATEGORY COLUMNS =====")
score_cols = [c for c in df.columns if c.endswith("_score")]
cat_cols = [c for c in df.columns if c.endswith("_category")]

print("score_cols:", score_cols)
print("cat_cols:", cat_cols)


print("\n===== MISSING SCORES =====")
print(df[score_cols].isna().sum())


print("\n===== CATEGORY COUNTS =====")
for c in cat_cols:
    print(f"\n{c}")
    print(df[c].value_counts(dropna=False))


print("\n===== SAMPLE USEFUL COLUMNS =====")
cols = [
    "instruction",
    "input",
    "output",
    "length_score",
    "length_category",
    "ppl_score",
    "ppl_category",
    "ifd_score",
    "ifd_category",
    "lexical_cluster_score",
    "lexical_cluster_category",
    "semantic_cluster_score",
    "semantic_cluster_category",
]

cols = [c for c in cols if c in df.columns]
print(df[cols].head(5).to_string(max_colwidth=120))
