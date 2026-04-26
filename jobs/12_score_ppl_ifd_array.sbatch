#!/bin/bash
#SBATCH --job-name=ppl-ifd
#SBATCH --output=logs/ppl-ifd-%A-%a.out
#SBATCH --error=logs/ppl-ifd-%A-%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
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
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_CACHE_ROOT="$PROJECT_DIR/.cache/vllm"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs outputs/curriculum_array datasets "$HF_HOME" "$HF_DATASETS_CACHE" "$VLLM_CACHE_ROOT"

echo "job=$SLURM_JOB_ID array_task=${SLURM_ARRAY_TASK_ID:-none} node=${SLURMD_NODENAME:-unknown}"
nvidia-smi || true

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
  bash -lc "cd '$PROJECT_DIR' && python3 -u scripts/12_score_ppl_ifd_chunk.py --config '$CONFIG'"
