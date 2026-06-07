#!/bin/bash
#SBATCH --job-name=sem-score
#SBATCH --output=logs/sem-score-%j.out
#SBATCH --error=logs/sem-score-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

# Avoid small compute-node /tmp. Some Singularity installations extract SIF
# into a temporary sandbox, which can easily exceed /tmp/job-* space.
export SINGULARITY_CACHEDIR="$PWD/.singularity/cache"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export SINGULARITY_TMPDIR="$PWD/.singularity/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT


PROJECT_DIR="$HOME/master-cluster-HSE"
IMAGE="$PROJECT_DIR/containers/current-sandbox"
CONFIG="${1:-configs/score_pythoncodes_cluster.yaml}"

cd "$PROJECT_DIR"

module purge

# Rocky Linux 9 container runtime setup.
# The old CentOS module name can be absent or exposed as legacy; prefer whichever runtime exists.
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

echo "container_runtime=$CONTAINER_RUNTIME"


export PROJECT_DIR="$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

mkdir -p logs outputs/curriculum_array

"$CONTAINER_RUNTIME" exec \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --env PROJECT_DIR="$PROJECT_DIR" \
  --env PYTHONPATH="$PYTHONPATH" \
  "$IMAGE" \
  bash -lc "cd '$PROJECT_DIR' && python3 -u scripts/15_score_semantic_from_embeddings.py --config '$CONFIG'"
