#!/usr/bin/env bash
#SBATCH --job-name=smoke-v100-085-qwen3
#SBATCH --output=logs/slurm/smoke-v100-vllm085-qwen3-%j.out
#SBATCH --error=logs/slurm/smoke-v100-vllm085-qwen3-%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$HOME/master-cluster-HSE}}"
if [[ ! -d "$PROJECT_DIR/containers" && -d "$HOME/master-cluster-HSE" ]]; then
  PROJECT_DIR="$HOME/master-cluster-HSE"
fi
cd "$PROJECT_DIR"

IMAGE="${RL_V100_VLLM085_IMAGE:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"

module purge || true
module load singularity/3.9.0

mkdir -p \
  logs/slurm \
  .singularity/cache \
  .singularity/tmp \
  .home \
  .cache/huggingface/datasets \
  .cache/vllm \
  .cache/torch \
  .cache/torch_extensions \
  .cache/torchinductor \
  .cache/triton \
  models datasets outputs logs

export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity/cache"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$SINGULARITY_TMPDIR"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

echo "===== JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-unknown} host=$(hostname)"
echo "project=$PROJECT_DIR"
echo "image=$IMAGE"
echo "pythonpath=/workspace"
echo "VLLM_SMOKE_MODEL=${VLLM_SMOKE_MODEL:-none}"

nvidia-smi || true

test -e "$IMAGE" || { echo "Missing sandbox: $IMAGE"; exit 1; }

# Critical: do NOT include /workspace/.python_metadata. It can contain stale
# dist-info records from old images and confuse importlib.metadata / Unsloth.
export PYTHONPATH=/workspace
unset SINGULARITYENV_PYTHONPATH || true

export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export VLLM_CACHE_ROOT=/workspace/.cache/vllm
export TORCH_HOME=/workspace/.cache/torch
export TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor
export TRITON_CACHE_DIR=/workspace/.cache/triton
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export PYTHONUNBUFFERED=1

SINGULARITY_BASE=(
  singularity exec --nv --cleanenv
  --bind "$PROJECT_DIR:/workspace"
  --pwd /workspace
  --env PROJECT_DIR=/workspace
  --env PYTHONPATH=/workspace
  --env HF_HOME=/workspace/.cache/huggingface
  --env HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
  --env TRANSFORMERS_CACHE=/workspace/.cache/huggingface
  --env VLLM_CACHE_ROOT=/workspace/.cache/vllm
  --env TORCH_HOME=/workspace/.cache/torch
  --env TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions
  --env TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor
  --env TRITON_CACHE_DIR=/workspace/.cache/triton
  --env TOKENIZERS_PARALLELISM=false
  --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE"
  --env TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE"
  --env HF_DATASETS_OFFLINE="$HF_DATASETS_OFFLINE"
  --env VLLM_ENABLE_CUDA_COMPATIBILITY=1
  --env VLLM_USE_V1="$VLLM_USE_V1"
  --env VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND"
  --env PYTHONUNBUFFERED=1
  "$IMAGE"
)


export VLLM_SMOKE_MODEL="${VLLM_SMOKE_MODEL:-/workspace/models/qwen3-4b}"
export SMOKE_GPU_MEMORY_UTILIZATION="${SMOKE_GPU_MEMORY_UTILIZATION:-0.85}"
export SMOKE_MAX_MODEL_LEN="${SMOKE_MAX_MODEL_LEN:-256}"
export SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-8}"
export SMOKE_LOGPROBS="${SMOKE_LOGPROBS:-5}"

echo "qwen3_model=$VLLM_SMOKE_MODEL"

srun "${SINGULARITY_BASE[@]}"   --env VLLM_SMOKE_MODEL="$VLLM_SMOKE_MODEL"   --env SMOKE_GPU_MEMORY_UTILIZATION="$SMOKE_GPU_MEMORY_UTILIZATION"   --env SMOKE_MAX_MODEL_LEN="$SMOKE_MAX_MODEL_LEN"   --env SMOKE_MAX_TOKENS="$SMOKE_MAX_TOKENS"   --env SMOKE_LOGPROBS="$SMOKE_LOGPROBS"   bash -lc '
set -euo pipefail
cd /workspace
python - <<"PY_PRECHECK"
import sys, importlib.metadata as md
import torch, torchvision
print("precheck python", sys.executable)
print("metadata torch", md.version("torch"))
print("import torch", torch.__version__, torch.version.cuda)
print("metadata torchvision", md.version("torchvision"))
print("import torchvision", torchvision.__version__)
assert md.version("torch").startswith("2.6.0"), md.version("torch")
assert torch.__version__.startswith("2.6.0"), torch.__version__
PY_PRECHECK
python -u scripts/smoke_v100_vllm085_stack.py --model "${VLLM_SMOKE_MODEL}"
'
