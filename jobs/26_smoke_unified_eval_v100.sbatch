#!/bin/bash
#SBATCH --job-name=smoke-unified-eval
#SBATCH --output=logs/slurm/smoke-unified-eval-%j.out
#SBATCH --error=logs/slurm/smoke-unified-eval-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
IMAGE="${RL_IMAGE:-$PROJECT_DIR/containers/sandboxes/rl-grpo}"
SELECTED_MODEL="${SELECTED_MODEL:-4b-instruct}"
cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity/cache"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" \
  logs/slurm outputs/eval_jobs/base logs/eval_jobs/base models datasets .home \
  .cache/huggingface .cache/vllm .cache/torch .cache/torch_extensions .cache/torchinductor .cache/triton
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

echo "===== JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-unknown} host=$(hostname)"
echo "image=$IMAGE SELECTED_MODEL=$SELECTED_MODEL"
nvidia-smi || true

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
echo "detected_gpu=$GPU_NAME"
if [[ "$GPU_NAME" == *"V100"* ]]; then
  # V100 is compute capability 7.0. The latest vLLM V1/CUDA stack can select
  # kernels/backends that are not compiled for sm_70. Prefer the legacy engine
  # and xFormers attention on V100 unless the caller overrides these explicitly.
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
else
  export VLLM_USE_V1="${VLLM_USE_V1:-}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-}"
fi
echo "VLLM_USE_V1=${VLLM_USE_V1:-<unset>}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-<unset>}"

test -e "$IMAGE" || { echo "Missing sandbox: $IMAGE"; exit 1; }

srun singularity exec --nv --cleanenv \
  --bind "$PROJECT_DIR:/workspace" \
  --env PROJECT_DIR=/workspace \
  --env PYTHONPATH=/workspace:/workspace/.python_metadata \
  --env MODELS_DIR=/workspace/models \
  --env DATASETS_DIR=/workspace/datasets \
  --env OUTPUTS_DIR=/workspace/outputs/eval_jobs/base \
  --env LOGS_DIR=/workspace/logs/eval_jobs/base \
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
  --env VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
  --env VLLM_USE_V1="${VLLM_USE_V1:-}" \
  --env VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-}" \
  --env SELECTED_MODEL="$SELECTED_MODEL" \
  --env MAX_TOKENS=${MAX_TOKENS:-2048} \
  --env GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88} \
  --env EVAL_ENABLE_LOGPROBS=${EVAL_ENABLE_LOGPROBS:-1} \
  --env EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-128} \
  --env PYTHONUNBUFFERED=1 \
  "$IMAGE" \
  bash -lc 'cd /workspace && python -u scripts/eval_lora_all.py --benchmarks mbpp,humaneval --limit 2 --include-base --only base'
