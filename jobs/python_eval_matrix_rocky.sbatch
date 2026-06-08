#!/usr/bin/env bash
#SBATCH --job-name=py-eval-matrix
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --array=1-1%1
#SBATCH --output=logs/slurm/python-eval-matrix-%A-%a.out
#SBATCH --error=logs/slurm/python-eval-matrix-%A-%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
IMAGE="${IMAGE:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"
MATRIX="${MATRIX:?Set MATRIX=path/to/eval_experiments.txt}"
PARTS_ROOT="${PARTS_ROOT:?Set PARTS_ROOT=outputs/.../eval_jobs}"
SELECTED_MODEL="${SELECTED_MODEL:-4b-instruct}"
BENCHMARKS="${EVAL_BENCHMARKS:-mbpp,humaneval}"
LIMIT="${EVAL_LIMIT:-}"

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

EXP_NAME="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MATRIX")"
if [[ -z "$EXP_NAME" ]]; then
  echo "No EXP_NAME for task ${SLURM_ARRAY_TASK_ID} in $MATRIX" >&2
  exit 2
fi

export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity/cache"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"

PART_OUTPUTS="/workspace/$PARTS_ROOT/$EXP_NAME"
PART_LOGS="/workspace/logs/pipeline_python/$EXP_NAME"

mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" logs/slurm "$PARTS_ROOT/$EXP_NAME" "logs/pipeline_python/$EXP_NAME" .cache/huggingface/datasets .cache/vllm .cache/torch .cache/triton .home
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

export SINGULARITYENV_PROJECT_DIR=/workspace
export SINGULARITYENV_PYTHONPATH=/workspace
export SINGULARITYENV_MODELS_DIR=/workspace/models
export SINGULARITYENV_DATASETS_DIR=/workspace/datasets
export SINGULARITYENV_OUTPUTS_DIR="$PART_OUTPUTS"
export SINGULARITYENV_LOGS_DIR="$PART_LOGS"
export SINGULARITYENV_HOME=/workspace/.home
export SINGULARITYENV_HF_HOME=/workspace/.cache/huggingface
export SINGULARITYENV_HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export SINGULARITYENV_TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export SINGULARITYENV_VLLM_CACHE_ROOT=/workspace/.cache/vllm
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_HF_DATASETS_OFFLINE=1
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_SELECTED_MODEL="$SELECTED_MODEL"
export SINGULARITYENV_MAX_TOKENS="${MAX_TOKENS:-2048}"
export SINGULARITYENV_EVAL_ENABLE_LOGPROBS="${EVAL_ENABLE_LOGPROBS:-1}"
export SINGULARITYENV_TORCHDYNAMO_DISABLE=1
export SINGULARITYENV_TORCH_COMPILE_DISABLE=1
export SINGULARITYENV_TORCHINDUCTOR_DISABLE=1
export SINGULARITYENV_PYTHONUNBUFFERED=1

LIMIT_ARGS=""
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS="--limit $LIMIT"
fi

if [[ "$EXP_NAME" == "base" ]]; then
  EVAL_ARGS="--benchmarks $BENCHMARKS $LIMIT_ARGS --include-base --only base"
else
  EVAL_ARGS="--benchmarks $BENCHMARKS $LIMIT_ARGS --only $EXP_NAME"
fi

echo "===== PY EVAL MATRIX ====="
echo "job=$SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID exp=$EXP_NAME"
echo "selected_model=$SELECTED_MODEL"
echo "parts_root=$PARTS_ROOT"
echo "eval_args=$EVAL_ARGS"
echo "runtime=$CONTAINER_RUNTIME"
nvidia-smi || true

srun "$CONTAINER_RUNTIME" exec --nv --cleanenv   --bind "$PROJECT_DIR:/workspace"   --pwd /workspace   "$IMAGE"   bash -lc "python -u scripts/eval_lora_all.py $EVAL_ARGS"
