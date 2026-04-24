#!/bin/bash
#SBATCH --job-name=clrl-check-env
#SBATCH --output=logs/check-env-%j.out
#SBATCH --error=logs/check-env-%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

PROJECT_DIR="$HOME/master-cluster-HSE"
IMAGE="$PROJECT_DIR/containers/current-sandbox"

cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

export PROJECT_DIR="$PROJECT_DIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export VLLM_CACHE_ROOT="$PROJECT_DIR/.cache/vllm"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

mkdir -p logs outputs datasets "$HF_HOME" "$VLLM_CACHE_ROOT"

echo "===== JOB INFO ====="
echo "job: $SLURM_JOB_ID"
echo "node: $SLURMD_NODENAME"
echo "date: $(date)"
echo "project: $PROJECT_DIR"
echo "image: $IMAGE"

echo "===== HOST GPU ====="
nvidia-smi || true

echo "===== CONTAINER CHECK ====="
singularity exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --env PROJECT_DIR="$PROJECT_DIR" \
  --env HF_HOME="$HF_HOME" \
  --env VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  --env TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
  "$IMAGE" \
  bash -lc "cd '$PROJECT_DIR' && python3 -u scripts/00_check_env.py"
