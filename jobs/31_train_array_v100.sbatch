#!/bin/bash
#SBATCH --job-name=train-cl-v100
#SBATCH --output=logs/slurm/train-cl-v100-%A-%a.out
#SBATCH --error=logs/slurm/train-cl-v100-%A-%a.err
#SBATCH --time=03:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-16%4

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
IMAGE="$PROJECT_DIR/containers/sif/unsloth-v100.sif"
CONFIG="${CONFIG:-configs/train/lora_pythoncodes_cl.yaml}"
MATRIX="${MATRIX:-outputs/cluster/train_runs.txt}"
cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

mkdir -p logs/slurm logs/train outputs/train_runs outputs/runs/train models datasets .home .cache/huggingface .cache/torch .cache/triton .cache/torchinductor

if [ ! -f "$MATRIX" ]; then
  singularity exec --cleanenv \
  --bind "$PWD:/workspace" \
  --env PYTHONPATH=/workspace \
  --env HOME=/workspace/.home \
  --env HF_HOME=/workspace/.cache/huggingface \
  containers/sif/vllm-v100.sif \
  bash -lc "cd /workspace && python -u scripts/cluster_make_run_matrices.py --config '$CONFIG'" 
fi

RUN_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MATRIX")
if [ -z "$RUN_NAME" ]; then
  echo "No run found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in $MATRIX"
  exit 1
fi

echo "===== JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} run=$RUN_NAME node=${SLURMD_NODENAME:-unknown} host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "image=$IMAGE"
echo "config=$CONFIG"
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
  --env CLUSTER_ARRAY_MODE=1 \
  --env PYTHONUNBUFFERED=1 \
  "$IMAGE" \
  bash -lc "cd /workspace && python -u scripts/train_cl_lora_all.py --config '$CONFIG' --only '$RUN_NAME' --local-files-only"
