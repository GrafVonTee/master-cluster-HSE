#!/bin/bash
#SBATCH --job-name=clingo-eval
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/clingo-eval-%A_%a.out
#SBATCH --error=logs/slurm/clingo-eval-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs/slurm .singularity/cache .singularity/tmp ".home"

export SINGULARITY_CACHEDIR="$PWD/.singularity/cache"
export SINGULARITY_TMPDIR="$PWD/.singularity/tmp/clingo_eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"

MODEL="${MODEL:?MODEL is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
DATASET="${DATASET:-/workspace/datasets/clingo/synthetic_v2}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-0}"
ADAPTER="${ADAPTER:-}"
MAX_TOKENS="${MAX_TOKENS:-512}"
GPU_UTIL="${GPU_UTIL:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"

SANDBOX="${SANDBOX:-containers/sandboxes/rl-grpo-v100-vllm085}"

CMD="python scripts/eval_clingo_vllm.py \
  --model '$MODEL' \
  --dataset '$DATASET' \
  --split '$SPLIT' \
  --out-dir '$OUT_DIR' \
  --limit '$LIMIT' \
  --max-tokens '$MAX_TOKENS' \
  --gpu-memory-utilization '$GPU_UTIL' \
  --max-model-len '$MAX_MODEL_LEN'"

if [[ -n "$ADAPTER" ]]; then
  CMD="$CMD --adapter '$ADAPTER'"
fi

singularity exec --cleanenv \
  --nv \
  --bind "$PWD:/workspace" \
  --pwd /workspace \
  --home "$PWD/.home:/workspace/.home" \
  "$SANDBOX" \
  bash -lc "set -euo pipefail; export PATH=/venv/main/bin:/usr/local/bin:/usr/bin:/bin:\$PATH; $CMD"

rm -rf "$SINGULARITY_TMPDIR" || true
