#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TAG="${RL_TAG:-master-cluster-hse/rl-grpo:latest}"
ARCHIVE="${RL_ARCHIVE:-containers/image_archives/rl-grpo.tar.gz}"

mkdir -p containers/image_archives

echo "[build] $TAG"
docker build -f containers/Dockerfile.rl-grpo -t "$TAG" .

echo "[save] $ARCHIVE"
docker save "$TAG" | gzip -c > "$ARCHIVE"

ls -lh "$ARCHIVE"
