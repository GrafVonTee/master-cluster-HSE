#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
cd "$PROJECT_DIR"

SANDBOX="${SANDBOX:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"

module purge 2>/dev/null || true
module load singularity/3.9.0 2>/dev/null || module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true

if command -v singularity >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v singularity)"
elif command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v apptainer)"
else
  echo "ERROR: no singularity/apptainer found" >&2
  exit 127
fi

if [ ! -d "$SANDBOX" ]; then
  echo "ERROR: missing sandbox: $SANDBOX" >&2
  echo "Set SANDBOX=/path/to/singularity_sandbox if it is stored elsewhere." >&2
  echo "Possible sandboxes:" >&2
  find "$PROJECT_DIR" "$HOME" -maxdepth 4 -type d \( -name "rl-grpo-v100-vllm085" -o -name "*sandbox*" -o -name "*vllm*" \) 2>/dev/null | head -50 >&2
  exit 2
fi

export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity/cache"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/login_py_$$"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"

mkdir -p \
  "$SINGULARITY_CACHEDIR" \
  "$SINGULARITY_TMPDIR" \
  "$PROJECT_DIR/.home" \
  "$PROJECT_DIR/.cache/huggingface/datasets" \
  "$PROJECT_DIR/.cache/vllm" \
  "$PROJECT_DIR/.cache/torch" \
  "$PROJECT_DIR/.cache/torch_extensions" \
  "$PROJECT_DIR/.cache/torchinductor" \
  "$PROJECT_DIR/.cache/triton" \
  "$PROJECT_DIR/logs/slurm"

trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

"$CONTAINER_RUNTIME" exec --cleanenv \
  --bind "$PROJECT_DIR:/workspace" \
  --pwd /workspace \
  --home "$PROJECT_DIR/.home:/workspace/.home" \
  --env PYTHONPATH=/workspace \
  --env HOME=/workspace/.home \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  --env TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
  --env VLLM_CACHE_ROOT=/workspace/.cache/vllm \
  --env TORCH_HOME=/workspace/.cache/torch \
  --env TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
  --env TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor \
  --env TRITON_CACHE_DIR=/workspace/.cache/triton \
  --env TOKENIZERS_PARALLELISM=false \
  --env UNSLOTH_DISABLE_STATISTICS=1 \
  "$SANDBOX" \
  bash -lc "set -euo pipefail; export PATH=/venv/main/bin:/usr/local/bin:/usr/bin:/bin:\$PATH; $*"
