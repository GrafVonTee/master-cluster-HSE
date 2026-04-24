import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


project_dir = Path(
    os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])
).resolve()

if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from src.data.prompt.pythoncodes import build_prompt
from src.data.curriculum.base import CurriculumPipeline
from src.data.curriculum.heuristics import LengthScorer
from src.data.curriculum.entropy import PPLScorer, IFDScorer
from src.data.curriculum.clustering import LexicalClusterScorer, SemanticClusterScorer
from src.data.curriculum.llm_judge import LLMJudgeScorer


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dtype_from_name(name):
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(name).lower(), torch.float16)


def needs_model(names):
    return any(x in {"ppl", "ifd", "semantic_cluster", "llm_judge"} for x in names)


def load_tokenizer(model_path, hf_home, offline):
    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=str(hf_home),
        local_files_only=offline,
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_path, cfg, hf_home, offline):
    dtype = dtype_from_name(cfg["model"].get("dtype", "float16"))

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=str(hf_home),
        local_files_only=offline,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def build_scorers(cfg, tokenizer, model):
    names = cfg["curriculum"]["scorers"]
    device = cfg["model"].get("device", "cuda")
    out = []

    for name in names:
        if name == "length":
            out.append(LengthScorer("length", tokenizer, build_prompt))

        elif name == "ppl":
            out.append(PPLScorer("ppl", tokenizer, model, build_prompt, device=device))

        elif name == "ifd":
            out.append(IFDScorer("ifd", tokenizer, model, build_prompt, device=device))

        elif name == "lexical_cluster":
            out.append(
                LexicalClusterScorer(
                    "lexical_cluster",
                    tokenizer,
                    n_clusters=int(cfg["curriculum"].get("lexical_clusters", 5)),
                )
            )

        elif name == "semantic_cluster":
            out.append(
                SemanticClusterScorer(
                    "semantic_cluster",
                    model,
                    tokenizer,
                    device=device,
                    n_clusters=int(cfg["curriculum"].get("semantic_clusters", 5)),
                )
            )

        elif name == "llm_judge":
            out.append(LLMJudgeScorer("llm_judge", model, tokenizer, device=device))

        else:
            raise ValueError(f"Unknown scorer: {name}")

    return out


def category_counts(ds, scorer_names):
    out = {}
    for name in scorer_names:
        col = f"{name}_category"
        if col not in ds.column_names:
            continue
        values = list(ds[col])
        out[col] = {k: values.count(k) for k in ["easy", "medium", "hard"]}
    return out


def score_stats(ds, scorer_names):
    out = {}
    for name in scorer_names:
        col = f"{name}_score"
        if col not in ds.column_names:
            continue

        arr = np.array(ds[col], dtype=float)
        valid = arr[~np.isnan(arr)]

        if len(valid) == 0:
            out[col] = {"count": 0, "nan_count": int(np.isnan(arr).sum())}
            continue

        out[col] = {
            "count": int(len(valid)),
            "nan_count": int(np.isnan(arr).sum()),
            "mean": float(valid.mean()),
            "std": float(valid.std()),
            "min": float(valid.min()),
            "p33": float(np.percentile(valid, 33)),
            "p66": float(np.percentile(valid, 66)),
            "max": float(valid.max()),
        }

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/score_pythoncodes_cl.yaml")
    args = parser.parse_args()

    cfg = read_yaml(args.config)

    offline = bool(cfg.get("runtime", {}).get("offline", True))
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    hf_home = Path(os.environ.get("HF_HOME", project_dir / ".cache" / "huggingface"))
    datasets_dir = project_dir / "datasets"

    hf_home.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg["dataset"]["name"]
    split = cfg["dataset"].get("split", "train")
    limit = cfg["dataset"].get("limit")

    model_path = cfg["model"].get("local_path") or cfg["model"]["name_or_path"]
    scorer_names = cfg["curriculum"]["scorers"]

    print("===== CONFIG =====")
    print("PROJECT_DIR:", project_dir)
    print("HF_HOME:", hf_home)
    print("offline:", offline)
    print("dataset:", dataset_name)
    print("split:", split)
    print("limit:", limit)
    print("model:", model_path)
    print("scorers:", scorer_names)

    print("\n===== LOAD DATASET =====")
    ds = load_dataset(
        dataset_name,
        split=split,
        cache_dir=str(datasets_dir),
    )

    if limit is not None:
        limit = int(limit)
        ds = ds.select(range(min(limit, len(ds))))

    print(ds)
    print("columns:", ds.column_names)

    print("\n===== LOAD TOKENIZER =====")
    tokenizer = load_tokenizer(model_path, hf_home, offline)

    model = None
    if needs_model(scorer_names):
        print("\n===== LOAD MODEL =====")
        model = load_model(model_path, cfg, hf_home, offline)

    print("\n===== BUILD SCORERS =====")
    scorers = build_scorers(cfg, tokenizer, model)
    print("built:", [s.name for s in scorers])

    print("\n===== RUN PIPELINE =====")
    pipeline = CurriculumPipeline(
        scorers=scorers,
        percentiles=cfg["curriculum"].get("percentiles", [33, 66]),
    )

    ds_scored = pipeline.process_dataset(
        ds,
        batch_size=int(cfg["curriculum"].get("batch_size", 4)),
    )

    print("\n===== SAVE OUTPUTS =====")
    out_cfg = cfg["output"]

    dataset_dir = project_dir / out_cfg["dataset_dir"]
    parquet_path = project_dir / out_cfg["parquet_path"]
    csv_path = project_dir / out_cfg["csv_path"]
    summary_path = project_dir / out_cfg["summary_path"]

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_dir.exists():
        import shutil
        shutil.rmtree(dataset_dir)

    ds_scored.save_to_disk(str(dataset_dir))
    ds_scored.to_parquet(str(parquet_path))

    preview_n = min(200, len(ds_scored))
    pd.DataFrame(ds_scored.select(range(preview_n))).to_csv(csv_path, index=False)

    summary = {
        "dataset": dataset_name,
        "split": split,
        "limit": limit,
        "num_rows": len(ds_scored),
        "columns": ds_scored.column_names,
        "scorers": scorer_names,
        "category_counts": category_counts(ds_scored, scorer_names),
        "score_stats": score_stats(ds_scored, scorer_names),
        "outputs": {
            "dataset_dir": str(dataset_dir),
            "parquet_path": str(parquet_path),
            "csv_path": str(csv_path),
            "summary_path": str(summary_path),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("\nDONE")


if __name__ == "__main__":
    main()
