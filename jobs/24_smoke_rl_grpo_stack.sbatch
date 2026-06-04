#!/bin/bash
#SBATCH --job-name=smoke-rl-grpo
#SBATCH --output=logs/slurm/smoke-rl-grpo-%j.out
#SBATCH --error=logs/slurm/smoke-rl-grpo-%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
IMAGE="${RL_IMAGE:-$PROJECT_DIR/containers/sandboxes/rl-grpo}"
cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity/cache"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" logs/slurm .home \
  .cache/huggingface .cache/vllm .cache/torch .cache/torch_extensions .cache/torchinductor .cache/triton
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

echo "===== JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-unknown} host=$(hostname)"
echo "image=$IMAGE"
echo "VLLM_SMOKE_MODEL=${VLLM_SMOKE_MODEL:-none}"
nvidia-smi || true

test -e "$IMAGE" || { echo "Missing image: $IMAGE"; exit 1; }

MODEL_ARG=""
if [[ -n "${VLLM_SMOKE_MODEL:-}" ]]; then
  MODEL_ARG="--model ${VLLM_SMOKE_MODEL}"
fi

srun singularity exec --nv --cleanenv \
  --bind "$PROJECT_DIR:/workspace" \
  --env PROJECT_DIR=/workspace \
  --env PYTHONPATH=/workspace:/workspace/.python_metadata \
  --env HOME=/workspace/.home \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  --env TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
  --env VLLM_CACHE_ROOT=/workspace/.cache/vllm \
  --env TORCH_HOME=/workspace/.cache/torch \
  --env TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
  --env TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor \
  --env TRITON_CACHE_DIR=/workspace/.cache/triton \
  --env TOKENIZERS_PARALLELISM=false \
  --env TORCH_CUDA_ARCH_LIST="7.0;8.9" \
  --env VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
  --env EVAL_ENABLE_LOGPROBS=1 \
  --env MAX_TOKENS=2048 \
  --env GPU_MEMORY_UTILIZATION=0.88 \
  --env SMOKE_GPU_MEMORY_UTILIZATION=${SMOKE_GPU_MEMORY_UTILIZATION:-0.85} \
  --env SMOKE_MAX_MODEL_LEN=${SMOKE_MAX_MODEL_LEN:-256} \
  --env PYTHONUNBUFFERED=1 \
  "$IMAGE" \
  bash -lc "cd /workspace && python -u scripts/smoke_rl_container.py --quick $MODEL_ARG"
