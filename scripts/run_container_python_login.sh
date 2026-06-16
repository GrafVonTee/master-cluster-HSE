#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/master-cluster-HSE}"

module purge
module load singularity/3.9.0 2>/dev/null || module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true

if command -v singularity >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v singularity)"
elif command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME="$(command -v apptainer)"
else
  echo "no singularity/apptainer" >&2
  exit 127
fi

export SINGULARITY_CACHEDIR="$PWD/.singularity/cache"
export SINGULARITY_TMPDIR="$PWD/.singularity/tmp/login_py_$$"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"

mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" .home
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

"$CONTAINER_RUNTIME" exec --cleanenv \
  --bind "$PWD:/workspace" \
  --pwd /workspace \
  --home "$PWD/.home:/workspace/.home" \
  containers/sandboxes/rl-grpo-v100-vllm085 \
  bash -lc "
    set -euo pipefail
    export PATH=/venv/main/bin:/usr/local/bin:/usr/bin:/bin:\$PATH
    $*
  "
