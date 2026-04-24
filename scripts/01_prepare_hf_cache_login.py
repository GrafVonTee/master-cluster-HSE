import argparse
import os
import sys
from pathlib import Path

import yaml
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


project_dir = Path(
    os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])
).resolve()

if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/score_pythoncodes_cl.yaml")
    args = parser.parse_args()

    cfg = read_yaml(args.config)

    hf_home = Path(os.environ.get("HF_HOME", project_dir / ".cache" / "huggingface"))
    datasets_dir = project_dir / "datasets"

    hf_home.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg["dataset"]["name"]
    split = cfg["dataset"].get("split", "train")

    model_path = cfg["model"].get("local_path") or cfg["model"]["name_or_path"]

    print("===== PREPARE HF CACHE =====")
    print("PROJECT_DIR:", project_dir)
    print("HF_HOME:", hf_home)
    print("dataset:", dataset_name)
    print("split:", split)
    print("model:", model_path)

    print("\n===== DOWNLOAD DATASET =====")
    ds = load_dataset(
        dataset_name,
        split=split,
        cache_dir=str(datasets_dir),
    )
    print(ds)
    print("columns:", ds.column_names)

    print("\n===== DOWNLOAD MODEL SNAPSHOT =====")
    snapshot_download(
        repo_id=model_path,
        cache_dir=str(hf_home),
        local_files_only=False,
        resume_download=True,
    )

    print("\n===== CHECK TOKENIZER =====")
    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=str(hf_home),
        local_files_only=False,
    )
    print("tokenizer:", type(tok).__name__)
    print("pad_token:", tok.pad_token)
    print("eos_token:", tok.eos_token)

    print("\nDONE")


if __name__ == "__main__":
    main()
