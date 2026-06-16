#!/bin/bash
#SBATCH --job-name=clingo-eval-matrix
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --array=1-1%1
#SBATCH --output=logs/slurm/clingo-eval-matrix-%A_%a.out
#SBATCH --error=logs/slurm/clingo-eval-matrix-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
SANDBOX="${SANDBOX:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"
MATRIX="${MATRIX:?Set MATRIX=path/to/eval_matrix.tsv}"
PARTS_ROOT="${PARTS_ROOT:-outputs/clingo_v3_100_small}"
DATASET="${DATASET:-/workspace/datasets/clingo/synthetic_v3_100}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-0}"
MAX_TOKENS="${MAX_TOKENS:-512}"
GPU_UTIL="${GPU_UTIL:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"

cd "$PROJECT_DIR"

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MATRIX")"
if [[ -z "$LINE" ]]; then
  echo "No eval matrix row for task ${SLURM_ARRAY_TASK_ID} in $MATRIX" >&2
  exit 2
fi

IFS=$'\t' read -r EXP_NAME MODEL ADAPTER <<< "$LINE"
ADAPTER="${ADAPTER:-}"
OUT_DIR="/workspace/${PARTS_ROOT}/${EXP_NAME}"

module purge
if command -v module >/dev/null 2>&1; then
  module load singularity/3.9.0 2>/dev/null || \
  module load singularity 2>/dev/null || \
  module load apptainer 2>/dev/null || true
fi

if command -v singularity >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v singularity)"
elif command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v apptainer)"
else
  echo "ERROR: neither singularity nor apptainer found on Rocky node" >&2
  echo "PATH=$PATH" >&2
  module list 2>&1 || true
  exit 127
fi

export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity/cache"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/clingo_eval_matrix_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"

mkdir -p \
  "$SINGULARITY_CACHEDIR" \
  "$SINGULARITY_TMPDIR" \
  logs/slurm \
  "$PARTS_ROOT/$EXP_NAME" \
  .home \
  .cache/huggingface/datasets \
  .cache/vllm \
  .cache/torch \
  .cache/torch_extensions \
  .cache/torchinductor \
  .cache/triton

cleanup() {
  rm -rf "$SINGULARITY_TMPDIR" || true
}
trap cleanup EXIT

echo "===== CLINGO EVAL MATRIX JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} node=${SLURMD_NODENAME:-unknown} host=$(hostname)"
echo "container_runtime=$CONTAINER_RUNTIME"
echo "sandbox=$SANDBOX"
echo "matrix=$MATRIX"
echo "experiment=$EXP_NAME"
echo "model=$MODEL"
echo "adapter=$ADAPTER"
echo "dataset=$DATASET"
echo "split=$SPLIT"
echo "limit=$LIMIT"
echo "out_dir=$OUT_DIR"
echo "gpu_util=$GPU_UTIL"
echo "max_model_len=$MAX_MODEL_LEN"
echo "max_tokens=$MAX_TOKENS"

nvidia-smi || true

test -d "$SANDBOX" || { echo "Missing sandbox: $SANDBOX" >&2; exit 1; }

CMD=(python -u scripts/eval_clingo_vllm.py
  --model "$MODEL"
  --dataset "$DATASET"
  --split "$SPLIT"
  --out-dir "$OUT_DIR"
  --limit "$LIMIT"
  --max-tokens "$MAX_TOKENS"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
)

if [[ -n "$ADAPTER" ]]; then
  CMD+=(--adapter "$ADAPTER")
fi

printf 'CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'

srun "$CONTAINER_RUNTIME" exec --nv --cleanenv \
  --bind "$PROJECT_DIR:/workspace" \
  --pwd /workspace \
  --home "$PROJECT_DIR/.home:/workspace/.home" \
  --env PYTHONPATH=/workspace \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  --env TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
  --env VLLM_CACHE_ROOT=/workspace/.cache/vllm \
  --env TORCH_HOME=/workspace/.cache/torch \
  --env TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
  --env TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor \
  --env TRITON_CACHE_DIR=/workspace/.cache/triton \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env UNSLOTH_DISABLE_STATISTICS=1 \
  "$SANDBOX" \
  bash -lc 'set -euo pipefail; export PATH=/venv/main/bin:/usr/local/bin:/usr/bin:/bin:$PATH; "$@"' bash "${CMD[@]}"
