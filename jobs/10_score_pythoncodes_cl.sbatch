#!/bin/bash
#SBATCH --job-name=score-pythoncodes-cl
#SBATCH --output=logs/score-pythoncodes-cl-%j.out
#SBATCH --error=logs/score-pythoncodes-cl-%j.err
#SBATCH --time=00:40:00
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

set -euo pipefail

PROJECT_DIR="$HOME/master-cluster-HSE"
IMAGE="$PROJECT_DIR/containers/current-sandbox"
CONFIG="${1:-configs/score_pythoncodes_cl.yaml}"

cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

export PROJECT_DIR="$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_CACHE_ROOT="$PROJECT_DIR/.cache/vllm"
export TOKENIZERS_PARALLELISM=false

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p \
  logs \
  datasets \
  outputs/curriculum \
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
echo "config: $CONFIG"
echo "PYTHONPATH: $PYTHONPATH"

echo "===== MODULES ====="
module list || true
echo "singularity: $(which singularity)"
singularity --version || true

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

if [ ! -f "scripts/10_score_pythoncodes_cl.py" ]; then
  echo "ERROR: script not found: scripts/10_score_pythoncodes_cl.py"
  ls -lah scripts || true
  exit 1
fi

echo "===== HOST GPU ====="
nvidia-smi || true

echo "===== RUN SCORING ====="
singularity exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --env PROJECT_DIR="$PROJECT_DIR" \
  --env PYTHONPATH="$PYTHONPATH" \
  --env HF_HOME="$HF_HOME" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
  --env TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
  --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  --env TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
  --env HF_DATASETS_OFFLINE="$HF_DATASETS_OFFLINE" \
  "$IMAGE" \
  bash -lc "cd '$PROJECT_DIR' && python3 -u scripts/10_score_pythoncodes_cl.py --config '$CONFIG'"

echo "===== OUTPUTS ====="
ls -lah outputs/curriculum || true
du -sh datasets/pythoncodes_cl_scored || true

echo "===== DONE ====="
