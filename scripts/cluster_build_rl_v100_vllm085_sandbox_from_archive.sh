#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
ARCHIVE="${ARCHIVE:-$PROJECT_DIR/containers/image_archives/rl-grpo-v100-vllm085.tar.gz}"
SANDBOX="${SANDBOX:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"
TMP_ROOT="${SINGULARITY_TMPDIR:-$PROJECT_DIR/.singularity/tmp/manual_v100_vllm085}"
CACHE_ROOT="${SINGULARITY_CACHEDIR:-$PROJECT_DIR/.singularity/cache}"

cd "$PROJECT_DIR"
module purge || true
module load singularity/3.9.0 || true

mkdir -p "$TMP_ROOT" "$CACHE_ROOT" "$(dirname "$SANDBOX")"
export SINGULARITY_TMPDIR="$TMP_ROOT"
export SINGULARITY_CACHEDIR="$CACHE_ROOT"

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
singularity build --sandbox "$SANDBOX" "docker-archive://$TMP_TAR"

rm -f "$TMP_TAR"

echo "===== DONE ====="
du -sh "$SANDBOX"
