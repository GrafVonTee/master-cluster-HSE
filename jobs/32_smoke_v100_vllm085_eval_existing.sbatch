#!/usr/bin/env bash
#SBATCH --job-name=smoke-v100-085-eval
#SBATCH --output=logs/slurm/smoke-v100-vllm085-eval-%j.out
#SBATCH --error=logs/slurm/smoke-v100-vllm085-eval-%j.err
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


SELECTED_MODEL="${SELECTED_MODEL:-4b-instruct}"
mkdir -p outputs/eval_jobs/v100-vllm085 logs/eval_jobs/v100-vllm085

echo "selected_model=$SELECTED_MODEL"

srun "${SINGULARITY_BASE[@]}"   --env MODELS_DIR=/workspace/models   --env DATASETS_DIR=/workspace/datasets   --env OUTPUTS_DIR=/workspace/outputs/eval_jobs/v100-vllm085   --env LOGS_DIR=/workspace/logs/eval_jobs/v100-vllm085   --env SELECTED_MODEL="$SELECTED_MODEL"   --env MAX_TOKENS="${MAX_TOKENS:-2048}"   --env GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"   --env EVAL_ENABLE_LOGPROBS="${EVAL_ENABLE_LOGPROBS:-1}"   --env EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"   bash -lc '
set -euo pipefail
cd /workspace
python - <<"PY_PRECHECK"
import importlib.metadata as md, torch
print("metadata torch", md.version("torch"))
print("import torch", torch.__version__, torch.version.cuda)
assert md.version("torch").startswith("2.6.0"), md.version("torch")
PY_PRECHECK
python -u scripts/eval_lora_all.py --benchmarks mbpp,humaneval --limit 2 --include-base --only base
'
