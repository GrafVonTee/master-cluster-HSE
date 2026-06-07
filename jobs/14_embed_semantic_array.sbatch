#!/bin/bash
#SBATCH --job-name=sem-emb
#SBATCH --output=logs/sem-emb-%A-%a.out
#SBATCH --error=logs/sem-emb-%A-%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
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
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs outputs/curriculum_array datasets "$HF_HOME" "$HF_DATASETS_CACHE"

echo "job=$SLURM_JOB_ID array_task=${SLURM_ARRAY_TASK_ID:-none} node=${SLURMD_NODENAME:-unknown}"
nvidia-smi || true

"$CONTAINER_RUNTIME" exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --env PROJECT_DIR="$PROJECT_DIR" \
  --env PYTHONPATH="$PYTHONPATH" \
  --env HF_HOME="$HF_HOME" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
  --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  --env TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
  --env HF_DATASETS_OFFLINE="$HF_DATASETS_OFFLINE" \
  "$IMAGE" \
  bash -lc "cd '$PROJECT_DIR' && python3 -u scripts/14_embed_semantic_chunk.py --config '$CONFIG'"
