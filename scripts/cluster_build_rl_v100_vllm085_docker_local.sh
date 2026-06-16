#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
IMAGE_NAME="${IMAGE_NAME:-master-cluster-hse/rl-grpo-v100-vllm085:latest}"
ARCHIVE="${ARCHIVE:-$PROJECT_DIR/containers/image_archives/rl-grpo-v100-vllm085.tar.gz}"

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$ARCHIVE")"

echo "===== BUILD DOCKER IMAGE ====="
echo "project=$PROJECT_DIR"
echo "image=$IMAGE_NAME"
echo "archive=$ARCHIVE"

docker build \
  -f containers/Dockerfile.rl-grpo-v100-vllm085 \
  -t "$IMAGE_NAME" \
  .

echo "===== QUICK IMAGE CHECK ====="
docker run --rm --gpus all "$IMAGE_NAME" nvidia-smi || true

echo "===== SAVE DOCKER ARCHIVE ====="
docker save "$IMAGE_NAME" | gzip -1 > "$ARCHIVE"
ls -lh "$ARCHIVE"
