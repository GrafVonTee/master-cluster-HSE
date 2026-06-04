#!/bin/bash
#SBATCH --job-name=merge-cl-v100
#SBATCH --output=logs/slurm/merge-cl-v100-%j.out
#SBATCH --error=logs/slurm/merge-cl-v100-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

IMAGE="$PROJECT_DIR/containers/sandboxes/vllm-v100"

export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity/cache"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

mkdir -p logs/slurm outputs/eval outputs/train_runs .home .cache/huggingface

echo "===== MERGE JOB ====="
echo "job=${SLURM_JOB_ID:-none}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "host=$(hostname)"
echo "image=$IMAGE"

test -e "$IMAGE" || { echo "Missing image: $IMAGE"; exit 1; }

singularity exec --cleanenv \
  --bind "$PROJECT_DIR:/workspace" \
  --env PROJECT_DIR=/workspace \
  --env PYTHONPATH=/workspace \
  --env MODELS_DIR=/workspace/models \
  --env DATASETS_DIR=/workspace/datasets \
  --env OUTPUTS_DIR=/workspace/outputs \
  --env LOGS_DIR=/workspace/logs \
  --env HOME=/workspace/.home \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env SELECTED_MODEL=4b \
  --env PYTHONUNBUFFERED=1 \
  "$IMAGE" \
  bash -lc 'cd /workspace && python -u scripts/cluster_merge_train_parts.py && python -u scripts/cluster_merge_eval_parts.py'
