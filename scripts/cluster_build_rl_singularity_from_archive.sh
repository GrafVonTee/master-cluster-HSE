#!/usr/bin/env bash
set -euo pipefail

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
export SINGULARITY_TMPDIR="$ROOT/.singularity/tmp/build_${NAME}_$$"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

ARCHIVE_TAR="$SINGULARITY_TMPDIR/${NAME}.tar"

echo "[gunzip transient] $ARCHIVE_GZ -> $ARCHIVE_TAR"
gunzip -c "$ARCHIVE_GZ" > "$ARCHIVE_TAR"

echo "[sandbox] $SANDBOX"
rm -rf "$SANDBOX"
singularity build --sandbox "$SANDBOX" "docker-archive://$ARCHIVE_TAR"

echo "[done]"
du -sh "$SANDBOX" || true
