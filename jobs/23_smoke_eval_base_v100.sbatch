#!/bin/bash
#SBATCH --job-name=smoke-eval-v100
#SBATCH --output=logs/slurm/smoke-eval-v100-%j.out
#SBATCH --error=logs/slurm/smoke-eval-v100-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
IMAGE="$PROJECT_DIR/containers/sif/vllm-v100.sif"
cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

mkdir -p logs/slurm outputs/eval_jobs/base logs/eval_jobs/base .home .cache/huggingface .cache/vllm

echo "===== JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-unknown} host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "image=$IMAGE"
nvidia-smi || true

test -f "$IMAGE" || { echo "Missing image: $IMAGE"; exit 1; }
test -d models/qwen3-4b || { echo "Missing model dir: models/qwen3-4b"; exit 1; }

srun singularity exec --nv --cleanenv \
  --bind "$PROJECT_DIR:/workspace" \
  --env PROJECT_DIR=/workspace \
  --env PYTHONPATH=/workspace \
  --env MODELS_DIR=/workspace/models \
  --env DATASETS_DIR=/workspace/datasets \
  --env OUTPUTS_DIR=/workspace/outputs/eval_jobs/base \
  --env LOGS_DIR=/workspace/logs/eval_jobs/base \
  --env HOME=/workspace/.home \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  --env TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
  --env VLLM_CACHE_ROOT=/workspace/.cache/vllm \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env SELECTED_MODEL=4b \
  --env MAX_TOKENS=2048 \
  --env EVAL_MAX_NEW_TOKENS=128 \
  --env PYTHONUNBUFFERED=1 \
  "$IMAGE" \
  bash -lc 'cd /workspace && python -u scripts/eval_lora_all.py --benchmarks mbpp --limit 2 --include-base --only base'
