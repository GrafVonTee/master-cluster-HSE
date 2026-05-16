#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAIN_TAG="${TRAIN_TAG:-master-cluster-hse/unsloth-v100:latest}"
EVAL_TAG="${EVAL_TAG:-master-cluster-hse/vllm-v100:latest}"

mkdir -p containers/image_archives

echo "[build] train image: $TRAIN_TAG"
docker build -f containers/Dockerfile.unsloth-v100 -t "$TRAIN_TAG" .

echo "[build] eval image: $EVAL_TAG"
docker build -f containers/Dockerfile.vllm-v100 -t "$EVAL_TAG" .

echo "[save] containers/image_archives/unsloth-v100.tar.gz"
docker save "$TRAIN_TAG" | gzip -c > containers/image_archives/unsloth-v100.tar.gz

echo "[save] containers/image_archives/vllm-v100.tar.gz"
docker save "$EVAL_TAG" | gzip -c > containers/image_archives/vllm-v100.tar.gz

echo "[done] archives are ready:"
ls -lh containers/image_archives/*.tar.gz
