#!/bin/bash
#SBATCH --job-name=make-cl-matrix
#SBATCH --output=logs/slurm/make-cl-matrix-%j.out
#SBATCH --error=logs/slurm/make-cl-matrix-%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

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


# Avoid small compute-node /tmp. Some Singularity installations extract SIF
# into a temporary sandbox, which can easily exceed /tmp/job-* space.
export SINGULARITY_CACHEDIR="$PWD/.singularity/cache"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export SINGULARITY_TMPDIR="$PWD/.singularity/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT


PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
CONFIG="${CONFIG:-configs/train/lora_pythoncodes_cl.yaml}"
cd "$PROJECT_DIR"
mkdir -p logs/slurm outputs/cluster
"$CONTAINER_RUNTIME" exec --cleanenv \
  --bind "$PWD:/workspace" \
  --env PYTHONPATH=/workspace \
  --env HOME=/workspace/.home \
  --env HF_HOME=/workspace/.cache/huggingface \
  containers/sif/vllm-v100.sif \
  bash -lc "cd /workspace && python -u scripts/cluster_make_run_matrices.py --config '$CONFIG'"
