#!/usr/bin/env python3
"""Download Qwen checkpoints into the repository model directory.

Default behavior stays compatible with the old helper: without arguments it
fetches Qwen3-14B.  For small-model experiments use, for example:

    python scripts/download_qwen14b.py --models 0_6b,1_7b

The script intentionally imports only huggingface_hub, not Unsloth/vLLM.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import snapshot_download


PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()

MODEL_ALIASES = {
    "0_6b": ("Qwen/Qwen3-0.6B", "qwen3-0.6b"),
    "0.6b": ("Qwen/Qwen3-0.6B", "qwen3-0.6b"),
    "1_7b": ("Qwen/Qwen3-1.7B", "qwen3-1.7b"),
    "1.7b": ("Qwen/Qwen3-1.7B", "qwen3-1.7b"),
    "4b": ("Qwen/Qwen3-4B", "qwen3-4b"),
    "4b-instruct": ("Qwen/Qwen3-4B-Instruct-2507", "qwen3-4b-instruct-2507"),
    "4b-thinking": ("Qwen/Qwen3-4B-Thinking-2507", "qwen3-4b-thinking-2507"),
    "8b": ("Qwen/Qwen3-8B", "qwen3-8b"),
    "14b": ("Qwen/Qwen3-14B", "qwen3-14b"),
}


def parse_models(raw_items: List[str]) -> List[str]:
    models: List[str] = []
    for raw in raw_items:
        for item in raw.split(","):
            item = item.strip()
            if item:
                models.append(item)
    return models or ["14b"]


def resolve_model(alias_or_repo: str, local_dir_root: Path) -> Tuple[str, Path]:
    key = alias_or_repo.strip()
    if key in MODEL_ALIASES:
        repo_id, folder = MODEL_ALIASES[key]
        return repo_id, local_dir_root / folder

    if "/" not in key:
        raise SystemExit(
            f"Unknown model alias: {key}. Available aliases: {', '.join(sorted(MODEL_ALIASES))}. "
            "For a custom Hugging Face repo, pass a full repo id like org/name."
        )

    folder = key.split("/", 1)[1].lower().replace("/", "-")
    return key, local_dir_root / folder


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="*",
        default=["14b"],
        help="Model aliases or HF repo ids. Supports comma-separated values. Default: 14b.",
    )
    parser.add_argument(
        "--local-dir-root",
        default=str(PROJECT_DIR / "models"),
        help="Directory where model folders like qwen3-0.6b will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME", str(PROJECT_DIR / ".cache" / "huggingface")),
        help="Hugging Face cache directory.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--allow-patterns", nargs="*", default=None)
    parser.add_argument("--ignore-patterns", nargs="*", default=None)
    args = parser.parse_args()

    local_dir_root = Path(args.local_dir_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    local_dir_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for alias_or_repo in parse_models(args.models):
        repo_id, local_dir = resolve_model(alias_or_repo, local_dir_root)
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"===== DOWNLOAD {alias_or_repo} =====", flush=True)
        print(f"repo_id={repo_id}", flush=True)
        print(f"local_dir={local_dir}", flush=True)
        print(f"cache_dir={cache_dir}", flush=True)

        path = snapshot_download(
            repo_id=repo_id,
            revision=args.revision,
            cache_dir=str(cache_dir),
            local_dir=str(local_dir),
            local_files_only=args.local_files_only,
            token=args.token,
            allow_patterns=args.allow_patterns,
            ignore_patterns=args.ignore_patterns,
        )
        config_json = local_dir / "config.json"
        tokenizer_json = local_dir / "tokenizer.json"
        result = {
            "alias_or_repo": alias_or_repo,
            "repo_id": repo_id,
            "snapshot_path": str(path),
            "local_dir": str(local_dir),
            "config_json_exists": config_json.exists(),
            "tokenizer_json_exists": tokenizer_json.exists(),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        results.append(result)

    print("===== SUMMARY =====")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
