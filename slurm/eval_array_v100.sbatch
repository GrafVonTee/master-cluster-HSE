#!/usr/bin/env bash
#SBATCH --job-name=qwen4b-cl-eval
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=1-17%4
#SBATCH --output=logs/slurm/eval_%A_%a.out
#SBATCH --error=logs/slurm/eval_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
EVAL_IMAGE="${EVAL_IMAGE:-$PROJECT_DIR/containers/sandboxes/vllm-v100}"
EVAL_MATRIX="${EVAL_MATRIX:-$PROJECT_DIR/outputs/cluster/eval_experiments.txt}"
EXP_NAME="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EVAL_MATRIX")"

if [[ -z "${EXP_NAME}" ]]; then
  echo "No experiment for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs/slurm" "$PROJECT_DIR/logs/eval" "$PROJECT_DIR/outputs/eval_jobs/$EXP_NAME" "$PROJECT_DIR/.cache"

echo "[eval] job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} exp=${EXP_NAME} node=$(hostname)"
echo "[eval] image=${EVAL_IMAGE}"
nvidia-smi || true

export SINGULARITYENV_PROJECT_DIR=/workspace
export SINGULARITYENV_MODELS_DIR=/workspace/models
export SINGULARITYENV_DATASETS_DIR=/workspace/datasets
# Important: make every eval array task write to a separate OUTPUTS_DIR to avoid CSV races.
export SINGULARITYENV_OUTPUTS_DIR="/workspace/outputs/eval_jobs/$EXP_NAME"
export SINGULARITYENV_LOGS_DIR=/workspace/logs
export SINGULARITYENV_HF_HOME=/workspace/.cache/huggingface
export SINGULARITYENV_HF_DATASETS_CACHE=/workspace/.cache/hf_datasets
export SINGULARITYENV_VLLM_CACHE_ROOT=/workspace/.cache/vllm
export SINGULARITYENV_XDG_CACHE_HOME=/workspace/.cache
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_SELECTED_MODEL="${SELECTED_MODEL:-4b-instruct}"
export SINGULARITYENV_MAX_TOKENS="${MAX_TOKENS:-2048}"
export SINGULARITYENV_GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
export SINGULARITYENV_EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1024}"

if [[ "$EXP_NAME" == "base" ]]; then
  CMD="python scripts/eval_lora_all.py --benchmarks mbpp,humaneval --include-base --only base"
else
  CMD="python scripts/eval_lora_all.py --benchmarks mbpp,humaneval --only '$EXP_NAME'"
fi

singularity exec --nv \
  --bind "$PROJECT_DIR:/workspace" \
  "$EVAL_IMAGE" \
  bash -lc "cd /workspace && $CMD"

echo "[eval] done exp=${EXP_NAME}"
nvidia-smi || true
