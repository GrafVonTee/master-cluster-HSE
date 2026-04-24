#!/bin/bash
#SBATCH --job-name=score-pythoncodes-cl
#SBATCH --output=logs/score-pythoncodes-cl-%j.out
#SBATCH --error=logs/score-pythoncodes-cl-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=70G

set -euo pipefail

PROJECT_DIR="$HOME/master-cluster-HSE"
IMAGE="$PROJECT_DIR/containers/current-sandbox"
CONFIG="${1:-configs/score_pythoncodes_cl.yaml}"

cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

export PROJECT_DIR="$PROJECT_DIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export VLLM_CACHE_ROOT="$PROJECT_DIR/.cache/vllm"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

mkdir -p logs outputs/curriculum datasets "$HF_HOME" "$VLLM_CACHE_ROOT"

echo "===== JOB INFO ====="
echo "job: $SLURM_JOB_ID"
echo "node: $SLURMD_NODENAME"
echo "date: $(date)"
echo "project: $PROJECT_DIR"
echo "image: $IMAGE"
echo "config: $CONFIG"

echo "===== HOST GPU ====="
nvidia-smi || true

echo "===== RUN SCORING ====="
singularity exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --env PROJECT_DIR="$PROJECT_DIR" \
  --env HF_HOME="$HF_HOME" \
  --env VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  --env TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
  "$IMAGE" \
  bash -lc "cd '$PROJECT_DIR' && python3 -u scripts/10_score_pythoncodes_cl.py --config '$CONFIG'"

echo "===== OUTPUTS ====="
ls -lah outputs/curriculum || true
du -sh datasets/pythoncodes_cl_scored || true
