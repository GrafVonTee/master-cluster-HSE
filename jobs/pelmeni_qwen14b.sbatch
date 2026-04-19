#!/bin/bash
#SBATCH --job-name=qwen14b-pelmeni
#SBATCH --output=logs/qwen14b-pelmeni-%j.out
#SBATCH --error=logs/qwen14b-pelmeni-%j.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

set -e

PROJECT_DIR="$HOME/master-cluster-HSE"
IMAGE="$PROJECT_DIR/containers/vllm-openai-0.8.5-sandbox"

cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export VLLM_CACHE_ROOT="$PROJECT_DIR/.cache/vllm"
export HF_HUB_OFFLINE=1

mkdir -p outputs .cache/huggingface .cache/vllm

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Workdir: $(pwd)"
echo "Date: $(date)"
echo

echo "===== HOST GPU INFO ====="
nvidia-smi
echo

echo "===== RUN CONTAINER SESSION ====="

singularity exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --env PROJECT_DIR="$PROJECT_DIR" \
  --env HF_HOME="$HF_HOME" \
  --env VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
  --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  "$IMAGE" \
  bash -lc '
    set -e

    cd "$PROJECT_DIR"

    echo "===== CONTAINER PYTHON INFO ====="
    which python3
    python3 -V
    echo

    echo "===== CONTAINER TORCH CHECK ====="
    python3 -c "import torch; print(\"torch:\", torch.__version__); print(\"cuda:\", torch.cuda.is_available()); print(\"devices:\", torch.cuda.device_count())"
    echo

    echo "===== CONTAINER VLLM CHECK ====="
    python3 -c "import vllm; print(\"vllm:\", vllm.__version__)"
    echo

    echo "===== RUN PELMENI SCRIPT ====="
    python3 -u scripts/pelmeni_qwen14b.py
  '
