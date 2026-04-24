#!/bin/bash
#SBATCH --job-name=merge-cl
#SBATCH --output=logs/merge-cl-%j.out
#SBATCH --error=logs/merge-cl-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

PROJECT_DIR="$HOME/master-cluster-HSE"
IMAGE="$PROJECT_DIR/containers/current-sandbox"
CONFIG="${1:-configs/score_pythoncodes_cluster.yaml}"

cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

export PROJECT_DIR="$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs outputs/curriculum_array datasets "$HF_HOME" "$HF_DATASETS_CACHE"

singularity exec \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --env PROJECT_DIR="$PROJECT_DIR" \
  --env PYTHONPATH="$PYTHONPATH" \
  --env HF_HOME="$HF_HOME" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  --env TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
  --env HF_DATASETS_OFFLINE="$HF_DATASETS_OFFLINE" \
  "$IMAGE" \
  bash -lc "cd '$PROJECT_DIR' && python3 -u scripts/19_merge_curriculum_scores.py --config '$CONFIG'"
