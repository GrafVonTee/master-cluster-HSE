#!/bin/bash
#SBATCH --job-name=clingo-v4-gen
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/clingo-v4-gen-%j.out
#SBATCH --error=logs/slurm/clingo-v4-gen-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
SANDBOX="${SANDBOX:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"
SOURCE="${SOURCE:-/workspace/datasets/clingo/synthetic_v3_100}"
OUT="${OUT:-/workspace/datasets/clingo/synthetic_v4}"
MODEL="${MODEL:-/workspace/models/qwen3-14b}"
SPLITS="${SPLITS:-train,validation,test}"
LIMIT_PER_SPLIT="${LIMIT_PER_SPLIT:-0}"
GPU_UTIL="${GPU_UTIL:-0.86}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_TOKENS="${MAX_TOKENS:-192}"
TEMPERATURE="${TEMPERATURE:-0.35}"
TOP_P="${TOP_P:-0.9}"
DRY_COPY="${DRY_COPY:-0}"

cd "$PROJECT_DIR"

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
  exit 127
fi

export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity/cache"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/clingo_v4_gen_${SLURM_JOB_ID}"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"

mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" logs/slurm datasets/clingo outputs/clingo_v4 .home .cache/huggingface/datasets .cache/vllm .cache/torch .cache/triton
trap 'rm -rf "$SINGULARITY_TMPDIR" || true' EXIT

ARGS=(python -u scripts/build_clingo_synthetic_v4_qwen.py
  --source "$SOURCE"
  --out "$OUT"
  --model "$MODEL"
  --splits "$SPLITS"
  --limit-per-split "$LIMIT_PER_SPLIT"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --local-files-only
)
if [[ "$DRY_COPY" == "1" ]]; then
  ARGS+=(--dry-copy)
fi

printf 'CMD:'
printf ' %q' "${ARGS[@]}"
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
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env UNSLOTH_DISABLE_STATISTICS=1 \
  "$SANDBOX" \
  bash -lc 'set -euo pipefail; export PATH=/venv/main/bin:/usr/local/bin:/usr/bin:/bin:$PATH; "$@"' bash "${ARGS[@]}"
