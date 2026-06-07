#!/bin/bash
#SBATCH --job-name=smoke-vllm-v100
#SBATCH --output=logs/slurm/smoke-vllm-v100-%j.out
#SBATCH --error=logs/slurm/smoke-vllm-v100-%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

set -euo pipefail

# Avoid small compute-node /tmp. Some Singularity installations extract SIF
# into a temporary sandbox, which can easily exceed /tmp/job-* space.
export SINGULARITY_CACHEDIR="$PWD/.singularity/cache"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export SINGULARITY_TMPDIR="$PWD/.singularity/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT


PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
IMAGE="$PROJECT_DIR/containers/sif/vllm-v100.sif"
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


mkdir -p logs/slurm .home .cache/huggingface .cache/vllm

echo "===== JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-unknown} host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "image=$IMAGE"
nvidia-smi || true

test -f "$IMAGE" || { echo "Missing image: $IMAGE"; exit 1; }

srun "$CONTAINER_RUNTIME" exec --nv --cleanenv \
  --bind "$PROJECT_DIR:/workspace" \
  --env PROJECT_DIR=/workspace \
  --env PYTHONPATH=/workspace \
  --env HOME=/workspace/.home \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  --env TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
  --env VLLM_CACHE_ROOT=/workspace/.cache/vllm \
  --env SELECTED_MODEL=4b \
  --env MAX_TOKENS=2048 \
  --env EVAL_MAX_NEW_TOKENS=128 \
  --env PYTHONUNBUFFERED=1 \
  "$IMAGE" \
  bash -lc 'cd /workspace && python -u - <<PY
import torch
print("torch:", torch.__version__, flush=True)
print("cuda:", torch.cuda.is_available(), flush=True)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, flush=True)
import vllm, transformers, peft, datasets, pandas
print("vllm:", vllm.__version__, flush=True)
print("transformers:", transformers.__version__, flush=True)
print("vllm smoke ok", flush=True)
PY'
