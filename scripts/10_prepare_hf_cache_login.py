import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.curriculum.cluster.io import read_yaml, hf_home, dataset_cache_dir, model_path_from_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/score_pythoncodes_cluster.yaml")
    args = parser.parse_args()

    cfg = read_yaml(args.config)

    dataset_name = cfg["dataset"]["name"]
    split = cfg["dataset"].get("split", "train")
    model_path = model_path_from_cfg(cfg)

    print("===== PREPARE HF CACHE =====")
    print("PROJECT_DIR:", PROJECT_DIR)
    print("HF_HOME:", hf_home())
    print("dataset_cache_dir:", dataset_cache_dir())
    print("dataset:", dataset_name)
    print("split:", split)
    print("model:", model_path)

    print("\n===== DATASET =====")
    ds = load_dataset(
        dataset_name,
        split=split,
        cache_dir=str(dataset_cache_dir()),
    )
    print(ds)
    print("columns:", ds.column_names)

    print("\n===== MODEL SNAPSHOT =====")
    snapshot_download(
        repo_id=model_path,
        cache_dir=str(hf_home()),
        local_files_only=False,
        resume_download=True,
    )

    print("\n===== TOKENIZER CHECK =====")
    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=str(hf_home()),
        local_files_only=False,
        use_fast=True,
    )

    print("tokenizer:", type(tok).__name__)
    print("pad_token:", tok.pad_token)
    print("eos_token:", tok.eos_token)

    print("\nDONE")


if __name__ == "__main__":
    main()
