#!/bin/bash
#SBATCH --job-name=verify-qwen3-8b
#SBATCH --output=logs/slurm/verify-qwen3-8b-%j.out
#SBATCH --error=logs/slurm/verify-qwen3-8b-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
IMAGE="${RL_IMAGE:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"

cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0 2>/dev/null || module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true

if command -v singularity >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v singularity)"
elif command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v apptainer)"
else
  echo "no singularity/apptainer" >&2
  exit 127
fi

export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity/cache"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/${SLURM_JOB_ID}"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"

mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" logs/slurm
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

echo "runtime=$CONTAINER_RUNTIME"
echo "image=$IMAGE"
du -sh models/qwen3-8b || true

srun "$CONTAINER_RUNTIME" exec --nv --cleanenv \
  --bind "$PROJECT_DIR:/workspace" \
  --pwd /workspace \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail

    export PYTHONPATH=/workspace
    export HF_HOME=/workspace/.cache/huggingface
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export SELECTED_MODEL=8b

    python - <<PY
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
import torch
import src.config as config

print("SELECTED_MODEL", config.SELECTED_MODEL)
print("MODEL_PATH", config.MODEL_PATH)
print("exists", Path(config.MODEL_PATH).exists())
print("config_json", (Path(config.MODEL_PATH) / "config.json").exists())

tok = AutoTokenizer.from_pretrained(config.MODEL_PATH, local_files_only=True, trust_remote_code=True)
cfg = AutoConfig.from_pretrained(config.MODEL_PATH, local_files_only=True, trust_remote_code=True)

print("tokenizer", tok.__class__.__name__)
print("model_type", cfg.model_type)
print("hidden_size", getattr(cfg, "hidden_size", None))
print("layers", getattr(cfg, "num_hidden_layers", None))
print("vocab_size", getattr(cfg, "vocab_size", None))
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
  '
