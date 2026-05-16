#!/usr/bin/env bash
#SBATCH --job-name=qwen4b-cl-merge
#SBATCH --partition=test
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/merge_%j.out
#SBATCH --error=logs/slurm/merge_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
EVAL_IMAGE="${EVAL_IMAGE:-$PROJECT_DIR/containers/sandboxes/vllm-v100}"
mkdir -p "$PROJECT_DIR/logs/slurm" "$PROJECT_DIR/outputs/eval"

export SINGULARITYENV_PROJECT_DIR=/workspace
export SINGULARITYENV_OUTPUTS_DIR=/workspace/outputs
export SINGULARITYENV_LOGS_DIR=/workspace/logs
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false

singularity exec \
  --bind "$PROJECT_DIR:/workspace" \
  "$EVAL_IMAGE" \
  bash -lc "cd /workspace && python scripts/cluster_merge_eval_parts.py --parts-root outputs/eval_jobs --out-dir outputs/eval"
