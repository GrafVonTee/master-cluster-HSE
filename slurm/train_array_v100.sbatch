#!/usr/bin/env bash
#SBATCH --job-name=qwen4b-cl-train
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=1-16%4
#SBATCH --output=logs/slurm/train_%A_%a.out
#SBATCH --error=logs/slurm/train_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
TRAIN_IMAGE="${TRAIN_IMAGE:-$PROJECT_DIR/containers/sandboxes/unsloth-v100}"
RUN_MATRIX="${RUN_MATRIX:-$PROJECT_DIR/outputs/cluster/train_runs.txt}"
RUN_NAME="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUN_MATRIX")"

if [[ -z "${RUN_NAME}" ]]; then
  echo "No run for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs/slurm" "$PROJECT_DIR/logs/train" "$PROJECT_DIR/outputs/train_runs" "$PROJECT_DIR/.cache"

echo "[train] job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} run=${RUN_NAME} node=$(hostname)"
echo "[train] image=${TRAIN_IMAGE}"
nvidia-smi || true

export SINGULARITYENV_PROJECT_DIR=/workspace
export SINGULARITYENV_MODELS_DIR=/workspace/models
export SINGULARITYENV_DATASETS_DIR=/workspace/datasets
export SINGULARITYENV_OUTPUTS_DIR=/workspace/outputs
export SINGULARITYENV_LOGS_DIR=/workspace/logs
export SINGULARITYENV_HF_HOME=/workspace/.cache/huggingface
export SINGULARITYENV_HF_DATASETS_CACHE=/workspace/.cache/hf_datasets
export SINGULARITYENV_VLLM_CACHE_ROOT=/workspace/.cache/vllm
export SINGULARITYENV_TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions
export SINGULARITYENV_XDG_CACHE_HOME=/workspace/.cache
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_SELECT_MODEL="${SELECT_MODEL:-}"
export SINGULARITYENV_SELECTED_MODEL="${SELECTED_MODEL:-4b-instruct}"
export SINGULARITYENV_MAX_TOKENS="${MAX_TOKENS:-2048}"
export SINGULARITYENV_FORCE_FP16=1
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Optional smoke override. Leave unset for full run.
export SINGULARITYENV_TRAIN_MAX_STEPS_OVERRIDE="${TRAIN_MAX_STEPS_OVERRIDE:-}"
export SINGULARITYENV_TRAIN_DATASET_LIMIT_OVERRIDE="${TRAIN_DATASET_LIMIT_OVERRIDE:-}"
export SINGULARITYENV_TRAIN_VAL_SIZE_OVERRIDE="${TRAIN_VAL_SIZE_OVERRIDE:-}"

singularity exec --nv \
  --bind "$PROJECT_DIR:/workspace" \
  "$TRAIN_IMAGE" \
  bash -lc "cd /workspace && python scripts/train_cl_lora_all.py --config configs/train/lora_pythoncodes_cl.yaml --only '$RUN_NAME'"

echo "[train] done run=${RUN_NAME}"
nvidia-smi || true
