#!/bin/bash
#SBATCH --job-name=clingo-train
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/clingo-train-%A_%a.out
#SBATCH --error=logs/slurm/clingo-train-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
SANDBOX="${SANDBOX:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"

CONFIG="${CONFIG:-configs/train/lora_clingo_synthetic.yaml}"
RUNS_FILE="${RUNS_FILE:-outputs/clingo/train_runs_4b.txt}"
RUN_NAME="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUNS_FILE" | tr -d '[:space:]')"

if [[ -z "$RUN_NAME" ]]; then
  echo "Empty RUN_NAME for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

cd "$PROJECT_DIR"

module purge
if command -v module >/dev/null 2>&1; then
  module load singularity/3.9.0 2>/dev/null ||   module load singularity 2>/dev/null ||   module load apptainer 2>/dev/null || true
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
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/clingo_train_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"

mkdir -p   "$SINGULARITY_CACHEDIR"   "$SINGULARITY_TMPDIR"   logs/slurm outputs/clingo .home   .cache/huggingface/datasets   .cache/vllm   .cache/torch   .cache/torch_extensions   .cache/torchinductor   .cache/triton

cleanup() {
  rm -rf "$SINGULARITY_TMPDIR" || true
}
trap cleanup EXIT

echo "===== CLINGO TRAIN JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} node=${SLURMD_NODENAME:-unknown} host=$(hostname)"
echo "container_runtime=$CONTAINER_RUNTIME"
echo "sandbox=$SANDBOX"
echo "config=$CONFIG"
echo "run_name=$RUN_NAME"
echo "TRAIN_MAX_STEPS_OVERRIDE=${TRAIN_MAX_STEPS_OVERRIDE:-}"
echo "TRAIN_DATASET_LIMIT_OVERRIDE=${TRAIN_DATASET_LIMIT_OVERRIDE:-}"
echo "TRAIN_VAL_SIZE_OVERRIDE=${TRAIN_VAL_SIZE_OVERRIDE:-}"

nvidia-smi || true

test -d "$SANDBOX" || { echo "Missing sandbox: $SANDBOX" >&2; exit 1; }
test -f "$CONFIG" || { echo "Missing config: $CONFIG" >&2; exit 1; }

srun "$CONTAINER_RUNTIME" exec --nv --cleanenv   --bind "$PROJECT_DIR:/workspace"   --pwd /workspace   --home "$PROJECT_DIR/.home:/workspace/.home"   --env PYTHONPATH=/workspace   --env HOME=/workspace/.home   --env HF_HOME=/workspace/.cache/huggingface   --env HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets   --env TRANSFORMERS_CACHE=/workspace/.cache/huggingface   --env VLLM_CACHE_ROOT=/workspace/.cache/vllm   --env TORCH_HOME=/workspace/.cache/torch   --env TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions   --env TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor   --env TRITON_CACHE_DIR=/workspace/.cache/triton   --env HF_HUB_OFFLINE=1   --env TRANSFORMERS_OFFLINE=1   --env HF_DATASETS_OFFLINE=1   --env TOKENIZERS_PARALLELISM=false   --env UNSLOTH_DISABLE_STATISTICS=1   --env CLUSTER_ARRAY_MODE=1   --env TRAIN_MAX_STEPS_OVERRIDE="${TRAIN_MAX_STEPS_OVERRIDE:-}"   --env TRAIN_DATASET_LIMIT_OVERRIDE="${TRAIN_DATASET_LIMIT_OVERRIDE:-}"   --env TRAIN_VAL_SIZE_OVERRIDE="${TRAIN_VAL_SIZE_OVERRIDE:-}"   "$SANDBOX"   bash -lc "set -euo pipefail; export PATH=/venv/main/bin:/usr/local/bin:/usr/bin:/bin:\$PATH; python -u scripts/train_clingo_lora_all.py --config '$CONFIG' --only '$RUN_NAME' --local-files-only --force"
