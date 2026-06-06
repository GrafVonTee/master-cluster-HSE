#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-master-cluster-hse/rl-grpo-v100-vllm085:latest}"

docker run --rm --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e PYTHONPATH=/workspace \
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
