#!/bin/bash
#SBATCH --job-name=clingo-v4-val
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/clingo-v4-val-%j.out
#SBATCH --error=logs/slurm/clingo-v4-val-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
SANDBOX="${SANDBOX:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"
SOURCE="${SOURCE:-/workspace/datasets/clingo/synthetic_v3_100}"
TARGET="${TARGET:-/workspace/datasets/clingo/synthetic_v4}"
OUT="${OUT:-/workspace/outputs/clingo_v4/validation}"
REQUIRE_CHANGED="${REQUIRE_CHANGED:-1}"
SKIP_SOLVER="${SKIP_SOLVER:-0}"

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
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/clingo_v4_val_${SLURM_JOB_ID}"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" logs/slurm outputs/clingo_v4 .home
trap 'rm -rf "$SINGULARITY_TMPDIR" || true' EXIT

ARGS=(python -u scripts/validate_clingo_synthetic_v4.py --source "$SOURCE" --target "$TARGET" --out "$OUT")
if [[ "$REQUIRE_CHANGED" == "1" ]]; then
  ARGS+=(--require-instruction-changed)
fi
if [[ "$SKIP_SOLVER" == "1" ]]; then
  ARGS+=(--skip-solver)
fi

srun "$CONTAINER_RUNTIME" exec --cleanenv \
  --bind "$PROJECT_DIR:/workspace" \
  --pwd /workspace \
  --home "$PROJECT_DIR/.home:/workspace/.home" \
  --env PYTHONPATH=/workspace \
  "$SANDBOX" \
  bash -lc 'set -euo pipefail; export PATH=/venv/main/bin:/usr/local/bin:/usr/bin:/bin:$PATH; "$@"' bash "${ARGS[@]}"
