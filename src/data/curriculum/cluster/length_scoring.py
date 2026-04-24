import pandas as pd
from tqdm import tqdm

from src.data.curriculum.cluster.io import (
    atomic_to_parquet,
    load_source_dataset,
    load_tokenizer_cached,
    out_root,
    require_columns,
)

from src.data.prompt.pythoncodes import build_prompt


def score_length(cfg: dict) -> str:
    out_path = out_root(cfg) / "length.parquet"

    if cfg.get("runtime", {}).get("skip_existing", True) and out_path.exists():
        print(f"skip existing: {out_path}", flush=True)
        return str(out_path)

    ds = load_source_dataset(cfg)
    require_columns(ds, ["instruction", "input", "output"])

    tokenizer = load_tokenizer_cached(cfg)

    rows = []

    for idx, row in tqdm(enumerate(ds), total=len(ds), desc="length"):
        text = build_prompt(row, tokenizer, train=True)["text"]
        score = len(tokenizer.encode(text, add_special_tokens=False))

        rows.append({
            "__idx": idx,
            "length_score": float(score),
        })

    df = pd.DataFrame(rows)
    atomic_to_parquet(df, out_path)

    print(f"saved: {out_path}", flush=True)
    return str(out_path)
