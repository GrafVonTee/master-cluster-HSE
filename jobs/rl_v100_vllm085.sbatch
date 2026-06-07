#!/usr/bin/env bash
#SBATCH --job-name=rl-v100-vllm085
#SBATCH --partition=rocky
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/rl-v100-vllm085-%x-%j.out
#SBATCH --error=logs/slurm/rl-v100-vllm085-%x-%j.err

set -euo pipefail

# Rocky Linux 9 container runtime setup.
# The old CentOS module name can be absent or exposed as legacy; prefer whichever runtime exists.
if command -v module >/dev/null 2>&1; then
  module load singularity/3.9.0 2>/dev/null || \
  module load singularity 2>/dev/null || \
  module load apptainer 2>/dev/null || true
fi

if command -v singularity >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v singularity)"
elif command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v apptainer)"
else
  echo "ERROR: neither singularity nor apptainer found on Rocky node" >&2
  echo "PATH=$PATH" >&2
  module list 2>&1 || true
  exit 127
fi

echo "container_runtime=$CONTAINER_RUNTIME"


cd "${SLURM_SUBMIT_DIR:-$PWD}"

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
IMAGE="${IMAGE:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"

TASK="${TASK:-stack}"                         # stack | qwen3 | train | eval
MODEL="${MODEL:-${VLLM_SMOKE_MODEL:-/workspace/models/qwen3-4b}}"
TRAIN_RUN="${TRAIN_RUN:-${TRAIN_SMOKE_RUN:-cl_length_staged}}"
EVAL_RUN="${EVAL_RUN:-$TRAIN_RUN}"
EVAL_INCLUDE_BASE="${EVAL_INCLUDE_BASE:-0}"

SMOKE_GPU_MEMORY_UTILIZATION="${SMOKE_GPU_MEMORY_UTILIZATION:-0.85}"
SMOKE_MAX_MODEL_LEN="${SMOKE_MAX_MODEL_LEN:-256}"
SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-8}"
SMOKE_LOGPROBS="${SMOKE_LOGPROBS:-5}"

TRAIN_MAX_STEPS_OVERRIDE="${TRAIN_MAX_STEPS_OVERRIDE:-1}"
TRAIN_DATASET_LIMIT_OVERRIDE="${TRAIN_DATASET_LIMIT_OVERRIDE:-300}"
TRAIN_VAL_SIZE_OVERRIDE="${TRAIN_VAL_SIZE_OVERRIDE:-50}"

EVAL_LIMIT="${EVAL_LIMIT:-2}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-mbpp,humaneval}"
GRPO_CONFIG="${GRPO_CONFIG:-configs/rl/grpo_pythoncodes.yaml}"
GRPO_RUN_NAME="${GRPO_RUN_NAME:-grpo_pythoncodes}"
GRPO_EVAL_LIMIT="${GRPO_EVAL_LIMIT:-2}"

mkdir -p "$PROJECT_DIR/logs/slurm"
mkdir -p "$PROJECT_DIR/.cache/huggingface/datasets" \
         "$PROJECT_DIR/.cache/vllm" \
         "$PROJECT_DIR/.cache/torch" \
         "$PROJECT_DIR/.cache/torch_extensions" \
         "$PROJECT_DIR/.cache/triton"

echo "===== JOB INFO ====="
echo "job=${SLURM_JOB_ID:-none}"
echo "task=$TASK"
echo "node=${SLURM_NODELIST:-none}"
echo "host=$(hostname)"
echo "pwd=$PWD"
echo "project=$PROJECT_DIR"
echo "image=$IMAGE"
echo "model=$MODEL"
echo "train_run=$TRAIN_RUN"
echo "eval_run=$EVAL_RUN"
echo "eval_include_base=$EVAL_INCLUDE_BASE"

test -d "$IMAGE" || { echo "ERROR: sandbox not found: $IMAGE" >&2; exit 2; }

nvidia-smi || true

# Keep repo importable, but do not expose stale package metadata.
export PYTHONPATH=/workspace
export SINGULARITYENV_PYTHONPATH=/workspace

# Disable Unsloth statistics/download probes. Required for offline/local model runs.
export UNSLOTH_DISABLE_STATISTICS=1
export SINGULARITYENV_UNSLOTH_DISABLE_STATISTICS=1

# Runtime caches inside mounted project dir.
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

# Force eager vLLM inside Unsloth GRPO on V100. Without this, Unsloth/vLLM 0.8.5
# may still initialize with enforce_eager=False and CUDA graph capture enabled.
export RL_FORCE_VLLM_EAGER_PATCH="${RL_FORCE_VLLM_EAGER_PATCH:-1}"
export RL_VLLM_DISABLE_CUSTOM_ALL_REDUCE="${RL_VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
export RL_VLLM_GPU_MEMORY_UTILIZATION="${RL_VLLM_GPU_MEMORY_UTILIZATION:-0.69}"

export SINGULARITYENV_RL_FORCE_VLLM_EAGER_PATCH="$RL_FORCE_VLLM_EAGER_PATCH"
export SINGULARITYENV_RL_VLLM_DISABLE_CUSTOM_ALL_REDUCE="$RL_VLLM_DISABLE_CUSTOM_ALL_REDUCE"
export SINGULARITYENV_RL_VLLM_GPU_MEMORY_UTILIZATION="$RL_VLLM_GPU_MEMORY_UTILIZATION"

# V100 vLLM/Triton stability knobs for GRPO.
export RL_VLLM_ENFORCE_EAGER="${RL_VLLM_ENFORCE_EAGER:-1}"
export RL_VLLM_DISABLE_CUSTOM_ALL_REDUCE="${RL_VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
export RL_VLLM_GPU_MEMORY_UTILIZATION="${RL_VLLM_GPU_MEMORY_UTILIZATION:-0.69}"
export RL_GRPO_VLLM_MAX_MODEL_LEN="${RL_GRPO_VLLM_MAX_MODEL_LEN:-2048}"

export SINGULARITYENV_RL_VLLM_ENFORCE_EAGER="$RL_VLLM_ENFORCE_EAGER"
export SINGULARITYENV_RL_VLLM_DISABLE_CUSTOM_ALL_REDUCE="$RL_VLLM_DISABLE_CUSTOM_ALL_REDUCE"
export SINGULARITYENV_RL_VLLM_GPU_MEMORY_UTILIZATION="$RL_VLLM_GPU_MEMORY_UTILIZATION"
export SINGULARITYENV_RL_GRPO_VLLM_MAX_MODEL_LEN="$RL_GRPO_VLLM_MAX_MODEL_LEN"

export VLLM_SMOKE_MODEL="$MODEL"
export SINGULARITYENV_VLLM_SMOKE_MODEL="$MODEL"

export SINGULARITYENV_SMOKE_GPU_MEMORY_UTILIZATION="$SMOKE_GPU_MEMORY_UTILIZATION"
export SINGULARITYENV_SMOKE_MAX_MODEL_LEN="$SMOKE_MAX_MODEL_LEN"
export SINGULARITYENV_SMOKE_MAX_TOKENS="$SMOKE_MAX_TOKENS"
export SINGULARITYENV_SMOKE_LOGPROBS="$SMOKE_LOGPROBS"

export SINGULARITYENV_TRAIN_MAX_STEPS_OVERRIDE="$TRAIN_MAX_STEPS_OVERRIDE"
export SINGULARITYENV_TRAIN_DATASET_LIMIT_OVERRIDE="$TRAIN_DATASET_LIMIT_OVERRIDE"
export SINGULARITYENV_TRAIN_VAL_SIZE_OVERRIDE="$TRAIN_VAL_SIZE_OVERRIDE"

export EVAL_ENABLE_LOGPROBS=1
export SINGULARITYENV_EVAL_ENABLE_LOGPROBS=1
export SINGULARITYENV_GRPO_CONFIG="$GRPO_CONFIG"
export SINGULARITYENV_GRPO_RUN_NAME="$GRPO_RUN_NAME"
export SINGULARITYENV_EVAL_RUN="$EVAL_RUN"
export SINGULARITYENV_EVAL_INCLUDE_BASE="$EVAL_INCLUDE_BASE"

run_in_container() {
  srun "$CONTAINER_RUNTIME" exec --nv \
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
for name in ["torch", "torchvision", "torchaudio", "vllm", "transformers", "trl", "peft", "unsloth", "unsloth_zoo", "xformers", "triton"]:
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

case "$TASK" in
  stack)
    precheck_versions
    echo "===== STACK SMOKE ====="
    run_in_container python -u scripts/smoke_v100_vllm085_stack.py
    ;;

  qwen3|vllm)
    precheck_versions
    echo "===== QWEN3 VLLM / LOGPROBS / ENTROPY SMOKE ====="
    run_in_container python -u scripts/smoke_v100_vllm085_stack.py --model "$MODEL"
    ;;

  train)
    precheck_versions
    echo "===== TRAIN SMOKE ====="
    echo "TRAIN_RUN=$TRAIN_RUN"
    echo "TRAIN_MAX_STEPS_OVERRIDE=$TRAIN_MAX_STEPS_OVERRIDE"
    echo "TRAIN_DATASET_LIMIT_OVERRIDE=$TRAIN_DATASET_LIMIT_OVERRIDE"
    echo "TRAIN_VAL_SIZE_OVERRIDE=$TRAIN_VAL_SIZE_OVERRIDE"

    run_in_container python -u scripts/train_cl_lora_all.py \
      --only "$TRAIN_RUN" \
      --local-files-only
    ;;

  eval)
    precheck_versions
    echo "===== EVAL SMOKE ====="
    echo "EVAL_BENCHMARKS=$EVAL_BENCHMARKS"
    echo "EVAL_LIMIT=$EVAL_LIMIT"

    if [ "$EVAL_INCLUDE_BASE" = "1" ]; then
      run_in_container python -u scripts/eval_lora_all.py \
        --benchmarks "$EVAL_BENCHMARKS" \
        --limit "$EVAL_LIMIT" \
        --include-base \
        --only base "$EVAL_RUN"
    else
      run_in_container python -u scripts/eval_lora_all.py \
        --benchmarks "$EVAL_BENCHMARKS" \
        --limit "$EVAL_LIMIT" \
        --only "$EVAL_RUN"
    fi
    ;;


  grpo_train)
    precheck_versions
    echo "===== GRPO TRAIN ====="
    echo "GRPO_CONFIG=$GRPO_CONFIG"
    echo "GRPO_RUN_NAME=$GRPO_RUN_NAME"
    run_in_container python -u scripts/rl_train_pythoncodes_grpo.py \
      --config "$GRPO_CONFIG" \
      --run-name "$GRPO_RUN_NAME" \
      --local-files-only
    ;;

  grpo_eval)
    precheck_versions
    echo "===== GRPO EVAL ====="
    run_in_container python -u scripts/rl_eval_grpo_adapter.py \
      --run-name "$GRPO_RUN_NAME" \
      --benchmarks "$EVAL_BENCHMARKS" \
      --limit "$GRPO_EVAL_LIMIT" \
      --include-base
    ;;

  *)
    echo "ERROR: unknown TASK=$TASK" >&2
    echo "Supported: TASK=stack | qwen3 | train | eval | grpo_train | grpo_eval" >&2
    exit 2
    ;;
esac

echo "===== TASK OK: $TASK ====="
