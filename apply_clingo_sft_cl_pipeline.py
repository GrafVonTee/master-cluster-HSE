#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path


def backup(path: Path) -> None:
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = path.with_suffix(path.suffix + f".bak_{stamp}")
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"backup: {bak}")


def write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup(path)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)
    print(f"wrote: {path}")


CLINGO_PROMPT = """\
\"\"\"
Prompt adapter for synthetic Clingo/ASP code-generation tasks.

Expected fields:
  instruction, facts, output/reference, difficulty, topic

The training target is assistant-only Clingo code.
\"\"\"

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

from src.config import DATASETS_DIR
from src.data.types import CodingTask

SYSTEM_PROMPT = (
    "You are an expert Answer Set Programming assistant. "
    "Return only valid clingo/ASP code. "
    "Do not explain. Do not use markdown."
)

STOP_TOKENS = ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]


def _non_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_dataset(
    disk_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    split: str = "train",
) -> Dataset:
    path = Path(disk_path) if disk_path is not None else DATASETS_DIR / "clingo" / "synthetic_v2_lora_train"
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        return dataset[split] if split in dataset else dataset["train"]
    return dataset


def build_messages(example: dict, train: bool = True) -> dict[str, list[dict[str, str]]]:
    instruction = _non_empty(example.get("instruction"))
    facts = _non_empty(example.get("facts"))

    user_content = f"Task:\\n{instruction}\\n\\nFacts:\\n{facts}\\n\\nWrite the clingo program."

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if train:
        target = _non_empty(example.get("reference")) or _non_empty(example.get("output"))
        messages.append({"role": "assistant", "content": target})

    return {"messages": messages}


def _apply_chat_template(tokenizer, messages, *, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def build_prompt(example: dict, tokenizer, train: bool = False) -> dict[str, str]:
    messages = build_messages(example, train=train)["messages"]
    return {"text": _apply_chat_template(tokenizer, messages, add_generation_prompt=not train)}


def build_tokenized_chat(
    example: dict,
    tokenizer,
    max_length: int,
    train: bool = True,
) -> dict[str, list[int]]:
    if not train:
        prompt = build_prompt(example, tokenizer=tokenizer, train=False)["text"]
        encoded = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    user_messages = build_messages(example, train=False)["messages"]
    full_messages = build_messages(example, train=True)["messages"]

    prompt_text = _apply_chat_template(tokenizer, user_messages, add_generation_prompt=True)
    full_text = _apply_chat_template(tokenizer, full_messages, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = list(input_ids)

    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    if all(x == -100 for x in labels) and labels:
        keep = min(64, len(labels))
        labels[-keep:] = input_ids[-keep:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def get_prepared_dataset(
    tokenizer=None,
    split: str = "train",
    disk_path: str | Path | None = None,
    parquet_path: str | Path | None = None,
    as_messages: bool = True,
) -> Dataset:
    dataset = get_dataset(disk_path=disk_path, parquet_path=parquet_path, split=split)

    if as_messages:
        return dataset.map(
            build_messages,
            fn_kwargs={"train": split == "train"},
            remove_columns=list(dataset.column_names),
            desc="Formatting clingo conversations",
        )

    if tokenizer is None:
        raise ValueError("tokenizer is required when as_messages=False")

    return dataset.map(
        build_prompt,
        fn_kwargs={"tokenizer": tokenizer, "train": split == "train"},
        remove_columns=list(dataset.column_names),
        desc="Formatting clingo prompts",
    )


def clingo_synthetic_to_task(row: dict, tokenizer) -> CodingTask:
    prompt_data = build_prompt(row, tokenizer, train=False)
    return CodingTask(
        prompt=prompt_data["text"],
        canonical_solution=row.get("reference", row.get("output", "")),
        tests=[],
        stop_tokens=STOP_TOKENS,
    )
"""


PREP_SCRIPT = """\
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

from datasets import Dataset, load_from_disk


def normalize_row(row: dict) -> dict:
    difficulty = str(row.get("difficulty", "")).strip()
    topic = str(row.get("topic", "")).strip()

    if difficulty not in {"easy", "medium", "hard"}:
        raise ValueError(f"Unexpected difficulty={difficulty!r} for task_id={row.get('task_id')}")

    reference = str(row.get("reference") or row.get("output") or "").strip()
    if not reference:
        raise ValueError(f"Empty reference for task_id={row.get('task_id')}")

    out = dict(row)
    out["output"] = reference
    out["reference"] = reference
    out["difficulty_category"] = difficulty
    out["topic_category"] = topic
    out["length_chars"] = len(reference)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="datasets/clingo/synthetic_v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="datasets/clingo/synthetic_v2_lora_train")
    args = parser.parse_args()

    ds = load_from_disk(args.source)
    if hasattr(ds, "keys"):
        ds = ds[args.split]

    rows = [normalize_row(dict(r)) for r in ds]
    out = Dataset.from_list(rows)
    out.save_to_disk(args.out)

    print(out)
    print("out", args.out)
    print("difficulty", Counter(out["difficulty_category"]))
    print("topic", Counter(out["topic_category"]))
    print("sample", out[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


CONFIG = """\
dataset:
  name: clingo_synthetic_v2
  disk_path: datasets/clingo/synthetic_v2_lora_train
  parquet_path:
  val_size: 24
  limit:
  seed: 42

curriculum:
  stages: [easy, medium, hard]
  distribution_stage_size: 160
  distribution_stages:
    - name: d1_80_15_5
      weights: {easy: 0.80, medium: 0.15, hard: 0.05}
    - name: d2_40_40_20
      weights: {easy: 0.40, medium: 0.40, hard: 0.20}
    - name: d3_20_20_60
      weights: {easy: 0.20, medium: 0.20, hard: 0.60}

model:
  base_model: /workspace/models/qwen3-4b-instruct-2507
  output_model_prefix: /workspace/models/qwen3-4b-instruct-2507-sft
  load_in_4bit: false

lora:
  r: 32
  alpha: 32
  dropout: 0
  bias: none
  use_gradient_checkpointing: unsloth
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  train_batch_size: 4
  gradient_steps: 4
  max_steps: 160
  stage_max_steps: 60
  warmup_steps: 10
  lr: 0.0001
  lr_scheduler_type: cosine
  optim: paged_adamw_8bit
  assistant_only_loss: true
  packing: false
  logging_steps: 5
  eval_steps: 20
  save_steps: 60
  save_total_limit: 2
  report_to: tensorboard
  dataloader_num_workers: 2
  dataloader_pin_memory: true
  max_grad_norm: 1.0
  seed: 42

runs:
  - name: sft_clingo
    category_col:
    schedule_type: plain

  - name: clingo_difficulty_staged
    category_col: difficulty_category
    schedule_type: staged

  - name: clingo_difficulty_cumulative
    category_col: difficulty_category
    schedule_type: cumulative

  - name: clingo_difficulty_distribution
    category_col: difficulty_category
    schedule_type: distribution
"""


JOB = """\
#!/bin/bash
#SBATCH --job-name=clingo-train
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/clingo-train-%A_%a.out
#SBATCH --error=logs/slurm/clingo-train-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
SANDBOX="${SANDBOX:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"

CONFIG="${CONFIG:-configs/train/lora_clingo_synthetic.yaml}"
RUNS_FILE="${RUNS_FILE:-outputs/clingo/train_runs_4b.txt}"
RUN_NAME="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUNS_FILE" | tr -d '[:space:]')"

if [[ -z "$RUN_NAME" ]]; then
  echo "Empty RUN_NAME for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

cd "$PROJECT_DIR"

module purge
if command -v module >/dev/null 2>&1; then
  module load singularity/3.9.0 2>/dev/null || \
  module load singularity 2>/dev/null || \
  module load apptainer 2>/dev/null || true
fi

if command -v singularity >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v singularity)"
elif command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v apptainer)"
else
  echo "ERROR: neither singularity nor apptainer found on Rocky node" >&2
  echo "PATH=$PATH" >&2
  module list 2>&1 || true
  exit 127
fi

export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity/cache"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/clingo_train_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"

mkdir -p \
  "$SINGULARITY_CACHEDIR" \
  "$SINGULARITY_TMPDIR" \
  logs/slurm outputs/clingo .home \
  .cache/huggingface/datasets \
  .cache/vllm \
  .cache/torch \
  .cache/torch_extensions \
  .cache/torchinductor \
  .cache/triton

cleanup() {
  rm -rf "$SINGULARITY_TMPDIR" || true
}
trap cleanup EXIT

echo "===== CLINGO TRAIN JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} node=${SLURMD_NODENAME:-unknown} host=$(hostname)"
echo "container_runtime=$CONTAINER_RUNTIME"
echo "sandbox=$SANDBOX"
echo "config=$CONFIG"
echo "run_name=$RUN_NAME"
echo "TRAIN_MAX_STEPS_OVERRIDE=${TRAIN_MAX_STEPS_OVERRIDE:-}"
echo "TRAIN_DATASET_LIMIT_OVERRIDE=${TRAIN_DATASET_LIMIT_OVERRIDE:-}"
echo "TRAIN_VAL_SIZE_OVERRIDE=${TRAIN_VAL_SIZE_OVERRIDE:-}"

nvidia-smi || true

test -d "$SANDBOX" || { echo "Missing sandbox: $SANDBOX" >&2; exit 1; }
test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 1; }

srun "$CONTAINER_RUNTIME" exec --nv --cleanenv \
  --bind "$PROJECT_DIR:/workspace" \
  --pwd /workspace \
  --home "$PROJECT_DIR/.home:/workspace/.home" \
  --env PYTHONPATH=/workspace \
  --env HOME=/workspace/.home \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  --env TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
  --env VLLM_CACHE_ROOT=/workspace/.cache/vllm \
  --env TORCH_HOME=/workspace/.cache/torch \
  --env TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
  --env TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor \
  --env TRITON_CACHE_DIR=/workspace/.cache/triton \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env UNSLOTH_DISABLE_STATISTICS=1 \
  --env CLUSTER_ARRAY_MODE=1 \
  --env TRAIN_MAX_STEPS_OVERRIDE="${TRAIN_MAX_STEPS_OVERRIDE:-}" \
  --env TRAIN_DATASET_LIMIT_OVERRIDE="${TRAIN_DATASET_LIMIT_OVERRIDE:-}" \
  --env TRAIN_VAL_SIZE_OVERRIDE="${TRAIN_VAL_SIZE_OVERRIDE:-}" \
  "$SANDBOX" \
  bash -lc "set -euo pipefail; export PATH=/venv/main/bin:/usr/local/bin:/usr/bin:/bin:\$PATH; python -u scripts/train_clingo_lora_all.py --config '$CONFIG' --only '$RUN_NAME' --local-files-only --force"
"""


def main() -> int:
    write(Path("src/data/prompt/clingo_synthetic.py"), CLINGO_PROMPT)
    write(Path("scripts/prepare_clingo_lora_dataset.py"), PREP_SCRIPT, executable=True)
    write(Path("configs/train/lora_clingo_synthetic.yaml"), CONFIG)
    write(Path("jobs/clingo_train_matrix_rocky.sbatch"), JOB, executable=True)

    source = Path("scripts/train_cl_lora_all.py")
    target = Path("scripts/train_clingo_lora_all.py")
    if not source.exists():
        raise SystemExit(f"Missing source trainer: {source}")

    text = source.read_text(encoding="utf-8")
    text = text.replace(
        "Container-side training script for pythoncodes CL experiments.",
        "Container-side training script for Clingo synthetic CL experiments.",
    )
    text = text.replace(
        "from src.data.prompt import pythoncodes_cl_scored",
        "from src.data.prompt import clingo_synthetic as pythoncodes_cl_scored",
    )
    text = text.replace(
        'parser.add_argument("--config", default="configs/train/lora_pythoncodes_cl.yaml")',
        'parser.add_argument("--config", default="configs/train/lora_clingo_synthetic.yaml")',
    )
    text = text.replace("Tokenizing scored pythoncodes chats", "Tokenizing clingo synthetic chats")
    text = text.replace("Loaded scored dataset:", "Loaded clingo dataset:")

    write(target, text, executable=True)

    print("\nNEXT:")
    print("  scripts/run_container_python_login.sh 'python scripts/prepare_clingo_lora_dataset.py'")
    print("  scripts/run_container_python_login.sh 'python scripts/train_clingo_lora_all.py --config configs/train/lora_clingo_synthetic.yaml --list-runs'")
    print("  scripts/run_container_python_login.sh 'TRAIN_MAX_STEPS_OVERRIDE=2 TRAIN_DATASET_LIMIT_OVERRIDE=80 TRAIN_VAL_SIZE_OVERRIDE=12 python scripts/train_clingo_lora_all.py --config configs/train/lora_clingo_synthetic.yaml --only sft_clingo --dry-run'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
