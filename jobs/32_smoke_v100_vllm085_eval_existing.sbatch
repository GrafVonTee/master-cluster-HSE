#!/usr/bin/env bash
#SBATCH --partition=test
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

#SBATCH --job-name=smoke-v100-vllm085-eval
#SBATCH --output=logs/slurm/smoke-v100-vllm085-eval-%j.out
#SBATCH --error=logs/slurm/smoke-v100-vllm085-eval-%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
IMAGE="${IMAGE:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"

mkdir -p "$PROJECT_DIR/logs/slurm"
mkdir -p "$PROJECT_DIR/.cache/huggingface/datasets" \
         "$PROJECT_DIR/.cache/vllm" \
         "$PROJECT_DIR/.cache/torch" \
         "$PROJECT_DIR/.cache/torch_extensions" \
         "$PROJECT_DIR/.cache/triton"

echo "===== JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none}"
echo "node=${SLURM_NODELIST:-none}"
echo "host=$(hostname)"
echo "pwd=$PWD"
echo "project=$PROJECT_DIR"
echo "image=$IMAGE"

test -d "$IMAGE" || { echo "ERROR: sandbox not found: $IMAGE" >&2; exit 2; }

nvidia-smi || true

# Do not include /workspace/.python_metadata. It can contain stale package metadata.
export PYTHONPATH=/workspace
export SINGULARITYENV_PYTHONPATH=/workspace

# Cache/env propagation for Singularity 3.9. Avoid `singularity exec --env ...`;
# option ordering differs by version and can be interpreted as the in-container command.
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export VLLM_CACHE_ROOT=/workspace/.cache/vllm
export TORCH_HOME=/workspace/.cache/torch
export TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions
export TRITON_CACHE_DIR=/workspace/.cache/triton

export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export SINGULARITYENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export SINGULARITYENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
export SINGULARITYENV_TORCH_HOME="$TORCH_HOME"
export SINGULARITYENV_TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR"
export SINGULARITYENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"

export SMOKE_GPU_MEMORY_UTILIZATION="${SMOKE_GPU_MEMORY_UTILIZATION:-0.85}"
export SMOKE_MAX_MODEL_LEN="${SMOKE_MAX_MODEL_LEN:-256}"
export SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-8}"
export SMOKE_LOGPROBS="${SMOKE_LOGPROBS:-5}"

export SINGULARITYENV_SMOKE_GPU_MEMORY_UTILIZATION="$SMOKE_GPU_MEMORY_UTILIZATION"
export SINGULARITYENV_SMOKE_MAX_MODEL_LEN="$SMOKE_MAX_MODEL_LEN"
export SINGULARITYENV_SMOKE_MAX_TOKENS="$SMOKE_MAX_TOKENS"
export SINGULARITYENV_SMOKE_LOGPROBS="$SMOKE_LOGPROBS"

run_in_container() {
  srun singularity exec --nv \
    --bind "$PROJECT_DIR:/workspace" \
    --pwd /workspace \
    "$IMAGE" \
    "$@"
}

precheck_versions() {
  echo "===== PRECHECK PACKAGE METADATA ====="
  run_in_container python - <<'PY'
import sys
import importlib.metadata as md

print("python", sys.executable)
for name in ["torch", "torchvision", "torchaudio", "vllm", "transformers", "trl", "unsloth", "unsloth_zoo", "torchao", "xformers", "triton"]:
    try:
        print(name, md.version(name))
    except Exception as e:
        print(name, "MISSING", repr(e))

import torch
print("torch.__version__", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("torch.__file__", torch.__file__)
print("torch.cuda.is_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
    print("arch_list", torch.cuda.get_arch_list())

import torchvision
print("torchvision.__version__", torchvision.__version__)
print("torchvision.__file__", torchvision.__file__)

assert md.version("torch").startswith("2.6.0"), md.version("torch")
assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torch.version.cuda == "12.4", torch.version.cuda
PY
}

precheck_versions

echo "===== EVAL SMOKE ====="
export EVAL_ENABLE_LOGPROBS=1
export SINGULARITYENV_EVAL_ENABLE_LOGPROBS=1

if [ -f scripts/eval_lora_all.py ]; then
  run_in_container python -u scripts/eval_lora_all.py \
    --benchmarks mbpp,humaneval \
    --limit 2 \
    --include-base \
    --only base
elif [ -f scripts/smoke_v100_vllm085_stack.py ]; then
  MODEL="${VLLM_SMOKE_MODEL:-/workspace/models/qwen3-4b}"
  export VLLM_SMOKE_MODEL="$MODEL"
  export SINGULARITYENV_VLLM_SMOKE_MODEL="$MODEL"
  echo "scripts/eval_lora_all.py not found; running vLLM smoke as eval-environment check with model=$MODEL"
  run_in_container python -u scripts/smoke_v100_vllm085_stack.py --model "$MODEL"
else
  echo "ERROR: no eval script and no smoke script found" >&2
  exit 2
fi
