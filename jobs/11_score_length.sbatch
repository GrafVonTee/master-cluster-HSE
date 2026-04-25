#!/bin/bash
#SBATCH --job-name=score-length
#SBATCH --output=logs/score-length-%j.out
#SBATCH --error=logs/score-length-%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

PROJECT_DIR="$HOME/master-cluster-HSE"
IMAGE="$PROJECT_DIR/containers/current-sandbox"
CONFIG="${1:-configs/score_pythoncodes_cluster.yaml}"

cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

export PROJECT_DIR="$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

export MODELS_DIR="$PROJECT_DIR/models"
export DATASETS_DIR="$PROJECT_DIR/datasets"
export OUTPUTS_DIR="$PROJECT_DIR/outputs"
export LOGS_DIR="$PROJECT_DIR/logs"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_CACHE_ROOT="$PROJECT_DIR/.cache/vllm"
export TOKENIZERS_PARALLELISM=false

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Надёжный проброс внутрь Singularity.
export SINGULARITYENV_PROJECT_DIR="$PROJECT_DIR"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"

export SINGULARITYENV_MODELS_DIR="$MODELS_DIR"
export SINGULARITYENV_DATASETS_DIR="$DATASETS_DIR"
export SINGULARITYENV_OUTPUTS_DIR="$OUTPUTS_DIR"
export SINGULARITYENV_LOGS_DIR="$LOGS_DIR"

export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export SINGULARITYENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
export SINGULARITYENV_TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM"

export SINGULARITYENV_HF_HUB_OFFLINE="$HF_HUB_OFFLINE"
export SINGULARITYENV_TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE"
export SINGULARITYENV_HF_DATASETS_OFFLINE="$HF_DATASETS_OFFLINE"

mkdir -p \
  logs \
  models \
  datasets \
  outputs/curriculum_array \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$VLLM_CACHE_ROOT"

echo "===== JOB INFO ====="
echo "job: ${SLURM_JOB_ID:-none}"
echo "node: ${SLURMD_NODENAME:-unknown}"
echo "project: $PROJECT_DIR"
echo "image: $IMAGE"
echo "config: $CONFIG"
echo "HF_HOME: $HF_HOME"
echo "DATASETS_DIR: $DATASETS_DIR"

echo "===== INPUT CHECK ====="

if [ ! -d "$IMAGE" ]; then
  echo "ERROR: sandbox not found: $IMAGE"
  ls -lah "$PROJECT_DIR/containers" || true
  exit 1
fi

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config not found: $CONFIG"
  ls -lah configs || true
  exit 1
fi

if [ ! -f "scripts/11_score_length.py" ]; then
  echo "ERROR: script not found: scripts/11_score_length.py"
  ls -lah scripts || true
  exit 1
fi

echo "===== CONTAINER ENV CHECK ====="
singularity exec \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  "$IMAGE" \
  bash -lc "echo PROJECT_DIR=\$PROJECT_DIR; echo HF_HOME=\$HF_HOME; echo DATASETS_DIR=\$DATASETS_DIR; python3 - <<'PY'
from src.config import PROJECT_DIR, HF_HOME, DATASETS_DIR, OUTPUTS_DIR
print('src.config PROJECT_DIR =', PROJECT_DIR)
print('src.config HF_HOME =', HF_HOME)
print('src.config DATASETS_DIR =', DATASETS_DIR)
print('src.config OUTPUTS_DIR =', OUTPUTS_DIR)
PY"

echo "===== RUN LENGTH SCORING ====="
singularity exec \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  "$IMAGE" \
  bash -lc "cd '$PROJECT_DIR' && python3 -u scripts/11_score_length.py --config '$CONFIG'"

echo "===== OUTPUTS ====="
ls -lah outputs/curriculum_array || true

echo "===== DONE ====="
