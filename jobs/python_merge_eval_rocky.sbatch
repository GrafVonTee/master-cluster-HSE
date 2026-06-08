#!/usr/bin/env bash
#SBATCH --job-name=py-merge-eval
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/python-merge-eval-%j.out
#SBATCH --error=logs/slurm/python-merge-eval-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
IMAGE="${IMAGE:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"
PARTS_ROOT="${PARTS_ROOT:?Set PARTS_ROOT=outputs/.../eval_jobs}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR=outputs/.../eval}"

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

export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity/cache"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/${SLURM_JOB_ID}"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"

mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" logs/slurm "$OUT_DIR"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

export SINGULARITYENV_PYTHONPATH=/workspace
export SINGULARITYENV_PYTHONUNBUFFERED=1

echo "===== PY MERGE EVAL ====="
echo "parts_root=$PARTS_ROOT"
echo "out_dir=$OUT_DIR"

srun "$CONTAINER_RUNTIME" exec --cleanenv   --bind "$PROJECT_DIR:/workspace"   --pwd /workspace   "$IMAGE"   bash -lc "python -u scripts/cluster_merge_eval_parts.py --parts-root '$PARTS_ROOT' --out-dir '$OUT_DIR'"
