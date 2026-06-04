#!/usr/bin/env bash
set -euo pipefail

# Build only a Singularity sandbox from a Docker archive.
# No SIF is produced. The temporary uncompressed Docker tar is placed under
# $PROJECT/.singularity/tmp and removed automatically, so /tmp and containers/sif
# do not get filled.
#
# Run on the cluster login node from ~/master-cluster-HSE after copying:
#   containers/image_archives/rl-grpo.tar.gz

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NAME="${RL_NAME:-rl-grpo}"
ARCHIVE_GZ="${RL_ARCHIVE_GZ:-containers/image_archives/${NAME}.tar.gz}"
SANDBOX="${RL_SANDBOX:-containers/sandboxes/${NAME}}"
REBUILD="${RL_REBUILD:-1}"

mkdir -p containers/image_archives containers/sandboxes .singularity/cache .singularity/tmp

if [[ ! -f "$ARCHIVE_GZ" ]]; then
  echo "Missing archive: $ARCHIVE_GZ" >&2
  exit 1
fi

export SINGULARITY_CACHEDIR="$ROOT/.singularity/cache"
export SINGULARITY_TMPDIR="$ROOT/.singularity/tmp/build_${NAME}_$$"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

TMP_ARCHIVE="$SINGULARITY_TMPDIR/${NAME}.tar"

if [[ -e "$SANDBOX" && "$REBUILD" == "1" ]]; then
  echo "[remove old sandbox] $SANDBOX"
  rm -rf "$SANDBOX"
fi

if [[ -e "$SANDBOX" ]]; then
  echo "[reuse existing sandbox] $SANDBOX"
else
  echo "[gunzip transient] $ARCHIVE_GZ -> $TMP_ARCHIVE"
  gunzip -c "$ARCHIVE_GZ" > "$TMP_ARCHIVE"

  echo "[build sandbox] $SANDBOX"
  singularity build --sandbox "$SANDBOX" "docker-archive://$TMP_ARCHIVE"
fi

# Remove stale products from the previous SIF-based workflow if they exist.
rm -f "containers/image_archives/${NAME}.tar"
if [[ "${RL_REMOVE_OLD_SIF:-0}" == "1" ]]; then
  rm -f "containers/sif/${NAME}.sif"
fi

echo "[done]"
du -sh "$SANDBOX" || true
