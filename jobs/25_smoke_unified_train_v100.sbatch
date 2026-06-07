#!/bin/bash
#SBATCH --job-name=smoke-unified-train
#SBATCH --output=logs/slurm/smoke-unified-train-%j.out
#SBATCH --error=logs/slurm/smoke-unified-train-%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
IMAGE="${RL_IMAGE:-$PROJECT_DIR/containers/sandboxes/rl-grpo}"
CONFIG="${CONFIG:-configs/train/lora_pythoncodes_cl_local_instruct.yaml}"
RUN_NAME="${RUN_NAME:-sft_pythoncodes}"
SELECTED_MODEL="${SELECTED_MODEL:-4b-instruct}"
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


export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity/cache"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" \
  logs/slurm logs/train outputs/train_runs outputs/runs/train models datasets .home \
  .cache/huggingface .cache/torch .cache/torch_extensions .cache/torchinductor .cache/triton
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

echo "===== JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-unknown} host=$(hostname)"
echo "image=$IMAGE"
echo "config=$CONFIG run=$RUN_NAME SELECTED_MODEL=$SELECTED_MODEL"
nvidia-smi || true

test -e "$IMAGE" || { echo "Missing sandbox: $IMAGE"; exit 1; }
test -f "$CONFIG" || { echo "Missing config: $CONFIG"; exit 1; }

if [[ ! -d datasets/pythoncodes_cl_scored && ! -f outputs/curriculum_array/pythoncodes_cl_scored.parquet ]]; then
  echo "Missing scored PythonCodes dataset: datasets/pythoncodes_cl_scored or outputs/curriculum_array/pythoncodes_cl_scored.parquet" >&2
  exit 1
fi

srun "$CONTAINER_RUNTIME" exec --nv --cleanenv \
  --bind "$PROJECT_DIR:/workspace" \
  --env PROJECT_DIR=/workspace \
  --env PYTHONPATH=/workspace \
  --env MODELS_DIR=/workspace/models \
  --env DATASETS_DIR=/workspace/datasets \
  --env OUTPUTS_DIR=/workspace/outputs \
  --env LOGS_DIR=/workspace/logs \
  --env HOME=/workspace/.home \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  --env TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
  --env TORCH_HOME=/workspace/.cache/torch \
  --env TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
  --env TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor \
  --env TRITON_CACHE_DIR=/workspace/.cache/triton \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env SELECTED_MODEL="$SELECTED_MODEL" \
  --env MAX_TOKENS=${MAX_TOKENS:-2048} \
  --env TRAIN_MAX_STEPS_OVERRIDE=${TRAIN_MAX_STEPS_OVERRIDE:-1} \
  --env TRAIN_DATASET_LIMIT_OVERRIDE=${TRAIN_DATASET_LIMIT_OVERRIDE:-300} \
  --env TRAIN_VAL_SIZE_OVERRIDE=${TRAIN_VAL_SIZE_OVERRIDE:-50} \
  --env CLUSTER_ARRAY_MODE=1 \
  --env PYTHONUNBUFFERED=1 \
  "$IMAGE" \
  bash -lc "cd /workspace && python -u scripts/train_cl_lora_all.py --config '$CONFIG' --only '$RUN_NAME' --local-files-only --force"
