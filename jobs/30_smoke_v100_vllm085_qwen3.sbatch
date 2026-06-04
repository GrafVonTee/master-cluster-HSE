#!/bin/bash
#SBATCH --job-name=smoke-v100-085-qwen3
#SBATCH --output=logs/slurm/smoke-v100-vllm085-qwen3-%j.out
#SBATCH --error=logs/slurm/smoke-v100-vllm085-qwen3-%j.err
#SBATCH --time=00:40:00
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

set -euo pipefail

export VLLM_SMOKE_MODEL="${VLLM_SMOKE_MODEL:-/workspace/models/qwen3-4b}"
export SMOKE_GPU_MEMORY_UTILIZATION="${SMOKE_GPU_MEMORY_UTILIZATION:-0.85}"
export SMOKE_MAX_MODEL_LEN="${SMOKE_MAX_MODEL_LEN:-256}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

bash "${BASH_SOURCE%/*}/29_smoke_v100_vllm085_stack.sbatch"
