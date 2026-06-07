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


# Run on the cluster login node from ~/master-cluster-HSE after copying
# containers/image_archives/rl-grpo.tar.gz from the local machine.
# This RL workflow intentionally builds only a sandbox and does not create SIF,
# because keeping both a sandbox and a SIF wastes cluster disk quota.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NAME="${RL_NAME:-rl-grpo}"
ARCHIVE_GZ="${RL_ARCHIVE_GZ:-containers/image_archives/${NAME}.tar.gz}"
SANDBOX="${RL_SANDBOX:-containers/sandboxes/${NAME}}"

mkdir -p containers/image_archives containers/sandboxes .singularity/cache .singularity/tmp

if [[ ! -f "$ARCHIVE_GZ" ]]; then
  echo "Missing archive: $ARCHIVE_GZ" >&2
  exit 1
fi

export SINGULARITY_CACHEDIR="$ROOT/.singularity/cache"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export SINGULARITY_TMPDIR="$ROOT/.singularity/tmp/build_${NAME}_$$"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

ARCHIVE_TAR="$SINGULARITY_TMPDIR/${NAME}.tar"

echo "[gunzip transient] $ARCHIVE_GZ -> $ARCHIVE_TAR"
gunzip -c "$ARCHIVE_GZ" > "$ARCHIVE_TAR"

echo "[sandbox] $SANDBOX"
rm -rf "$SANDBOX"
"$CONTAINER_RUNTIME" build --sandbox "$SANDBOX" "docker-archive://$ARCHIVE_TAR"

echo "[done]"
du -sh "$SANDBOX" || true
