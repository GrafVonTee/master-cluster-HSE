#!/usr/bin/env bash
#SBATCH --job-name=py-train-matrix
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --array=1-1%1
#SBATCH --output=logs/slurm/python-train-matrix-%A-%a.out
#SBATCH --error=logs/slurm/python-train-matrix-%A-%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
IMAGE="${IMAGE:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"
MATRIX="${MATRIX:?Set MATRIX=path/to/train_runs.txt}"
CONFIG="${CONFIG:-configs/train/lora_pythoncodes_cl.yaml}"
SELECTED_MODEL="${SELECTED_MODEL:-4b-instruct}"

cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0 2>/dev/null || module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true

if command -v singularity >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v singularity)"
elif command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v apptainer)"
else
  echo "ERROR: neither singularity nor apptainer found" >&2
  exit 127
fi

RUN_NAME="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MATRIX")"
if [[ -z "$RUN_NAME" ]]; then
  echo "No RUN_NAME for task ${SLURM_ARRAY_TASK_ID} in $MATRIX" >&2
  exit 2
fi

export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity/cache"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"

mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" logs/slurm .cache/huggingface/datasets .cache/vllm .cache/torch .cache/torch_extensions .cache/triton .home
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

if [[ "${ALLOW_TRAIN_OVERRIDES:-0}" != "1" ]]; then
  unset TRAIN_MAX_STEPS_OVERRIDE TRAIN_DATASET_LIMIT_OVERRIDE TRAIN_VAL_SIZE_OVERRIDE
fi
unset SINGULARITYENV_TRAIN_MAX_STEPS_OVERRIDE SINGULARITYENV_TRAIN_DATASET_LIMIT_OVERRIDE SINGULARITYENV_TRAIN_VAL_SIZE_OVERRIDE

export SINGULARITYENV_PROJECT_DIR=/workspace
export SINGULARITYENV_PYTHONPATH=/workspace
export SINGULARITYENV_MODELS_DIR=/workspace/models
export SINGULARITYENV_DATASETS_DIR=/workspace/datasets
export SINGULARITYENV_OUTPUTS_DIR=/workspace/outputs
export SINGULARITYENV_LOGS_DIR=/workspace/logs
export SINGULARITYENV_HOME=/workspace/.home
export SINGULARITYENV_HF_HOME=/workspace/.cache/huggingface
export SINGULARITYENV_HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export SINGULARITYENV_TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export SINGULARITYENV_VLLM_CACHE_ROOT=/workspace/.cache/vllm
export SINGULARITYENV_TORCH_HOME=/workspace/.cache/torch
export SINGULARITYENV_TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions
export SINGULARITYENV_TRITON_CACHE_DIR=/workspace/.cache/triton
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_HF_DATASETS_OFFLINE=1
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_SELECTED_MODEL="$SELECTED_MODEL"
export SINGULARITYENV_MAX_TOKENS="${MAX_TOKENS:-2048}"
export SINGULARITYENV_UNSLOTH_DISABLE_STATISTICS=1
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_TRAIN_MAX_STEPS_OVERRIDE="${TRAIN_MAX_STEPS_OVERRIDE:-}"
export SINGULARITYENV_TRAIN_DATASET_LIMIT_OVERRIDE="${TRAIN_DATASET_LIMIT_OVERRIDE:-}"
export SINGULARITYENV_TRAIN_VAL_SIZE_OVERRIDE="${TRAIN_VAL_SIZE_OVERRIDE:-}"

echo "===== PY TRAIN MATRIX ====="
echo "job=$SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID run=$RUN_NAME"
echo "selected_model=$SELECTED_MODEL"
echo "cpus=${SLURM_CPUS_PER_TASK:-unknown}"
echo "cuda_visible=${CUDA_VISIBLE_DEVICES:-none}"
echo "runtime=$CONTAINER_RUNTIME"
echo "image=$IMAGE"
echo "ALLOW_TRAIN_OVERRIDES=${ALLOW_TRAIN_OVERRIDES:-0}"
echo "TRAIN_MAX_STEPS_OVERRIDE=${TRAIN_MAX_STEPS_OVERRIDE:-}"
echo "TRAIN_DATASET_LIMIT_OVERRIDE=${TRAIN_DATASET_LIMIT_OVERRIDE:-}"
echo "TRAIN_VAL_SIZE_OVERRIDE=${TRAIN_VAL_SIZE_OVERRIDE:-}"
nvidia-smi || true

srun "$CONTAINER_RUNTIME" exec --nv --cleanenv   --bind "$PROJECT_DIR:/workspace"   --pwd /workspace   "$IMAGE"   bash -lc "python -u scripts/train_cl_lora_all.py --config '$CONFIG' --only '$RUN_NAME' --local-files-only --force"
