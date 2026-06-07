#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-master-cluster-hse/rl-grpo-v100-vllm085:latest}"

docker run --rm --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e RL_VLLM_ENFORCE_EAGER="${RL_VLLM_ENFORCE_EAGER:-1}" \
  -e RL_VLLM_DISABLE_CUSTOM_ALL_REDUCE="${RL_VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}" \
  -e RL_VLLM_GPU_MEMORY_UTILIZATION="${RL_VLLM_GPU_MEMORY_UTILIZATION:-0.69}" \
  -e RL_GRPO_VLLM_MAX_MODEL_LEN="${RL_GRPO_VLLM_MAX_MODEL_LEN:-2048}" \
  -e UNSLOTH_DISABLE_STATISTICS=1 \
  -e SINGULARITYENV_UNSLOTH_DISABLE_STATISTICS=1 \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
  -e HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  -e VLLM_CACHE_ROOT=/workspace/.cache/vllm \
  -e TORCH_HOME=/workspace/.cache/torch \
  -e TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
  -e TRITON_CACHE_DIR=/workspace/.cache/triton \
  "$IMAGE" \
  "$@"
