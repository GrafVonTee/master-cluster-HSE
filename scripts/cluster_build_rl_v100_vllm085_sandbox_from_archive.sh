#!/usr/bin/env bash
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


PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
ARCHIVE="${ARCHIVE:-$PROJECT_DIR/containers/image_archives/rl-grpo-v100-vllm085.tar.gz}"
SANDBOX="${SANDBOX:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"
TMP_ROOT="${SINGULARITY_TMPDIR:-$PROJECT_DIR/.singularity/tmp/manual_v100_vllm085}"
CACHE_ROOT="${SINGULARITY_CACHEDIR:-$PROJECT_DIR/.singularity/cache}"

cd "$PROJECT_DIR"
module purge || true

mkdir -p "$TMP_ROOT" "$CACHE_ROOT" "$(dirname "$SANDBOX")"
export SINGULARITY_TMPDIR="$TMP_ROOT"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
export SINGULARITY_CACHEDIR="$CACHE_ROOT"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"

if [[ ! -f "$ARCHIVE" ]]; then
  echo "Missing archive: $ARCHIVE" >&2
  exit 1
fi

rm -rf "$SANDBOX"
TMP_TAR="$TMP_ROOT/rl-grpo-v100-vllm085.tar"
rm -f "$TMP_TAR"

echo "===== UNPACK DOCKER ARCHIVE ====="
echo "archive=$ARCHIVE"
echo "tmp_tar=$TMP_TAR"
gzip -dc "$ARCHIVE" > "$TMP_TAR"

echo "===== BUILD SINGULARITY SANDBOX ONLY ====="
echo "sandbox=$SANDBOX"
"$CONTAINER_RUNTIME" build --sandbox "$SANDBOX" "docker-archive://$TMP_TAR"

rm -f "$TMP_TAR"

echo "===== DONE ====="
du -sh "$SANDBOX"
