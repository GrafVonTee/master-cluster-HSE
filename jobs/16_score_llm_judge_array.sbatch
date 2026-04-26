#!/bin/bash
#SBATCH --job-name=llm-judge
#SBATCH --output=logs/llm-judge-%A-%a.out
#SBATCH --error=logs/llm-judge-%A-%a.err
#SBATCH --time=00:30:00
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
echo "job=$SLURM_JOB_ID"
echo "array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "project=$PROJECT_DIR"
echo "config=$CONFIG"
echo "HF_HOME=$HF_HOME"

echo "===== GPU ====="
nvidia-smi || true

echo "===== RUN LLM JUDGE ====="
singularity exec --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  "$IMAGE" \
  bash -lc "cd '$PROJECT_DIR' && python3 -u scripts/16_score_llm_judge_chunk.py --config '$CONFIG'"

echo "===== OUTPUTS ====="
ls -lah outputs/curriculum_array/llm_judge_chunks || true

echo "===== DONE ====="
