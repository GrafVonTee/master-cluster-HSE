#!/bin/bash
#SBATCH --job-name=smoke-train-v100
#SBATCH --output=logs/slurm/smoke-train-v100-%j.out
#SBATCH --error=logs/slurm/smoke-train-v100-%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
IMAGE="$PROJECT_DIR/containers/sif/unsloth-v100.sif"
CONFIG="${CONFIG:-configs/train/lora_pythoncodes_cl.yaml}"
RUN_NAME="${RUN_NAME:-sft_pythoncodes}"
cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

mkdir -p logs/slurm logs/train outputs/train_runs outputs/runs/train models datasets .home .cache/huggingface .cache/torch .cache/triton .cache/torchinductor

echo "===== JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-unknown} host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "run=$RUN_NAME config=$CONFIG image=$IMAGE"
nvidia-smi || true

test -f "$IMAGE" || { echo "Missing image: $IMAGE"; exit 1; }
test -d models/qwen3-4b || { echo "Missing model dir: models/qwen3-4b"; exit 1; }
test -d datasets/pythoncodes_cl_scored || { echo "Missing dataset: datasets/pythoncodes_cl_scored"; exit 1; }

srun singularity exec --nv --cleanenv \
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
  --env TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor \
  --env TRITON_CACHE_DIR=/workspace/.cache/triton \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env SELECTED_MODEL=4b \
  --env MAX_TOKENS=2048 \
  --env TRAIN_MAX_STEPS_OVERRIDE=1 \
  --env TRAIN_DATASET_LIMIT_OVERRIDE=300 \
  --env TRAIN_VAL_SIZE_OVERRIDE=50 \
  --env CLUSTER_ARRAY_MODE=1 \
  --env PYTHONUNBUFFERED=1 \
  "$IMAGE" \
  bash -lc "cd /workspace && python -u scripts/train_cl_lora_all.py --config '$CONFIG' --only '$RUN_NAME' --local-files-only --force"
