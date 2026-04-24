import pandas as pd
from tqdm import tqdm

from src.data.curriculum.cluster.io import (
    atomic_to_parquet,
    load_source_dataset,
    load_tokenizer_cached,
    out_root,
    require_columns,
)

from src.data.curriculum.clustering import LexicalClusterScorer


def score_lexical(cfg: dict) -> str:
    out_path = out_root(cfg) / "lexical.parquet"

    if cfg.get("runtime", {}).get("skip_existing", True) and out_path.exists():
        print(f"skip existing: {out_path}", flush=True)
        return str(out_path)

    ds = load_source_dataset(cfg)
    require_columns(ds, ["output"])

    tokenizer = load_tokenizer_cached(cfg)

    scorer = LexicalClusterScorer(
        "lexical_cluster",
        tokenizer,
        n_clusters=int(cfg["lexical"].get("n_clusters", 10)),
    )

    scorer.fit(ds)

    rows = []
    batch_size = 512

    for start in tqdm(range(0, len(ds), batch_size), desc="lexical"):
        end = min(start + batch_size, len(ds))
        batch = ds.select(range(start, end))
        batch_dict = {col: batch[col] for col in batch.column_names}

        scores = scorer.score(batch_dict)

        for i, score in enumerate(scores):
            rows.append({
                "__idx": start + i,
                "lexical_cluster_score": float(score),
            })

    df = pd.DataFrame(rows)
    atomic_to_parquet(df, out_path)

    print(f"saved: {out_path}", flush=True)
    return str(out_path)
