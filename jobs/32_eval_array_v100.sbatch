#!/bin/bash
#SBATCH --job-name=eval-cl-v100
#SBATCH --output=logs/slurm/eval-cl-v100-%A-%a.out
#SBATCH --error=logs/slurm/eval-cl-v100-%A-%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-17%4

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
IMAGE="$PROJECT_DIR/containers/sif/vllm-v100.sif"
CONFIG="${CONFIG:-configs/train/lora_pythoncodes_cl.yaml}"
MATRIX="${MATRIX:-outputs/cluster/eval_experiments.txt}"
BENCHMARKS="${EVAL_BENCHMARKS:-mbpp,humaneval}"
NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1024}"
LIMIT="${EVAL_LIMIT:-}"
cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

mkdir -p logs/slurm outputs/eval_jobs logs/eval_jobs models datasets .home .cache/huggingface .cache/vllm

if [ ! -f "$MATRIX" ]; then
  python3 -u scripts/cluster_make_run_matrices.py --config "$CONFIG"
fi

EXP_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MATRIX")
if [ -z "$EXP_NAME" ]; then
  echo "No experiment found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in $MATRIX"
  exit 1
fi

EXPERIMENT_OUTPUTS="/workspace/outputs/eval_jobs/$EXP_NAME"
EXPERIMENT_LOGS="/workspace/logs/eval_jobs/$EXP_NAME"
mkdir -p "outputs/eval_jobs/$EXP_NAME" "logs/eval_jobs/$EXP_NAME"

LIMIT_ARGS=""
if [ -n "$LIMIT" ]; then
  LIMIT_ARGS="--limit $LIMIT"
fi

if [ "$EXP_NAME" = "base" ]; then
  EVAL_ARGS="--benchmarks $BENCHMARKS $LIMIT_ARGS --include-base --only base"
else
  EVAL_ARGS="--benchmarks $BENCHMARKS $LIMIT_ARGS --only $EXP_NAME"
fi

echo "===== JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} experiment=$EXP_NAME node=${SLURMD_NODENAME:-unknown} host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "image=$IMAGE"
echo "benchmarks=$BENCHMARKS max_new_tokens=$NEW_TOKENS limit=${LIMIT:-none}"
echo "EVAL_ARGS=$EVAL_ARGS"
nvidia-smi || true

test -f "$IMAGE" || { echo "Missing image: $IMAGE"; exit 1; }
test -d models/qwen3-4b || { echo "Missing model dir: models/qwen3-4b"; exit 1; }
if [ "$EXP_NAME" != "base" ]; then
  test -f "models/qwen3-4b-sft-$EXP_NAME/adapter_config.json" || { echo "Missing adapter for $EXP_NAME"; exit 1; }
fi

srun singularity exec --nv --cleanenv \
  --bind "$PROJECT_DIR:/workspace" \
  --env PROJECT_DIR=/workspace \
  --env PYTHONPATH=/workspace \
  --env MODELS_DIR=/workspace/models \
  --env DATASETS_DIR=/workspace/datasets \
  --env OUTPUTS_DIR="$EXPERIMENT_OUTPUTS" \
  --env LOGS_DIR="$EXPERIMENT_LOGS" \
  --env HOME=/workspace/.home \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  --env TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
  --env VLLM_CACHE_ROOT=/workspace/.cache/vllm \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env SELECTED_MODEL=4b \
  --env MAX_TOKENS=2048 \
  --env EVAL_MAX_NEW_TOKENS="$NEW_TOKENS" \
  --env PYTHONUNBUFFERED=1 \
  "$IMAGE" \
  bash -lc "cd /workspace && python -u scripts/eval_lora_all.py $EVAL_ARGS"
