#!/bin/bash
#SBATCH --job-name=clrl-check-env
#SBATCH --output=logs/check-env-%j.out
#SBATCH --error=logs/check-env-%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=test

set -euo pipefail

PROJECT_DIR="$HOME/master-cluster-HSE"
IMAGE="$PROJECT_DIR/containers/current-sandbox"

cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

export PROJECT_DIR="$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_CACHE_ROOT="$PROJECT_DIR/.cache/vllm"
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  logs \
  outputs \
  datasets \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$VLLM_CACHE_ROOT"

echo "===== JOB INFO ====="
echo "job: ${SLURM_JOB_ID:-none}"
echo "node: ${SLURMD_NODENAME:-unknown}"
echo "hostname: $(hostname)"
echo "date: $(date)"
echo "project: $PROJECT_DIR"
echo "image: $IMAGE"
echo "pwd: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

echo "===== MODULES ====="
module list || true
echo "singularity: $(which singularity)"
singularity --version || true

echo "===== PROJECT TREE CHECK ====="
ls -lah "$PROJECT_DIR"
echo
ls -lah "$PROJECT_DIR/scripts" || true
echo
ls -lah "$PROJECT_DIR/src" || true
echo

if [ ! -d "$IMAGE" ]; then
  echo "ERROR: sandbox image directory not found:"
  echo "$IMAGE"
  echo
  echo "Check:"
  echo "  ls -lah $PROJECT_DIR/containers"
  exit 1
fi

if [ ! -f "$PROJECT_DIR/scripts/00_check_env.py" ]; then
  echo "ERROR: script not found:"
  echo "$PROJECT_DIR/scripts/00_check_env.py"
  echo
  echo "Available scripts:"
  ls -lah "$PROJECT_DIR/scripts" || true
  exit 1
fi

if [ ! -d "$PROJECT_DIR/src" ]; then
  echo "ERROR: src directory not found:"
  echo "$PROJECT_DIR/src"
  exit 1
fi

echo "===== HOST GPU ====="
nvidia-smi || true

echo "===== CONTAINER BASIC CHECK ====="
singularity exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --env PROJECT_DIR="$PROJECT_DIR" \
  --env PYTHONPATH="$PYTHONPATH" \
  --env HF_HOME="$HF_HOME" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
  --env TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
  "$IMAGE" \
  bash -lc "python3 --version && which python3 && nvidia-smi || true"

echo "===== CONTAINER PYTHON CHECK ====="
singularity exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --env PROJECT_DIR="$PROJECT_DIR" \
  --env PYTHONPATH="$PYTHONPATH" \
  --env HF_HOME="$HF_HOME" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
  --env TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
  "$IMAGE" \
  bash -lc "cd '$PROJECT_DIR' && python3 -u scripts/00_check_env.py"

echo "===== DONE ====="
