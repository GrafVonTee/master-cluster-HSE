#!/bin/bash
#SBATCH --job-name=build-clrl-sandbox
#SBATCH --output=logs/build-sandbox-%j.out
#SBATCH --error=logs/build-sandbox-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=test

set -euo pipefail

PROJECT_DIR="$HOME/master-cluster-HSE"
CONTAINERS_DIR="$PROJECT_DIR/containers"

ARCHIVE_ZST="$CONTAINERS_DIR/clrl-v100-2026-04-24.docker.tar.zst"
ARCHIVE_TAR="$CONTAINERS_DIR/clrl-v100-2026-04-24.docker.tar"

SANDBOX="$CONTAINERS_DIR/clrl-v100-2026-04-24-sandbox"
CURRENT="$CONTAINERS_DIR/current-sandbox"

cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

mkdir -p logs "$CONTAINERS_DIR" "$PROJECT_DIR/.cache/singularity" "$PROJECT_DIR/.tmp/singularity"

export SINGULARITY_CACHEDIR="$PROJECT_DIR/.cache/singularity"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.tmp/singularity"
export TMPDIR="$PROJECT_DIR/.tmp/singularity"

echo "===== INFO ====="
echo "job: ${SLURM_JOB_ID:-none}"
echo "node: ${SLURMD_NODENAME:-unknown}"
echo "project: $PROJECT_DIR"
echo "containers: $CONTAINERS_DIR"
echo "sandbox: $SANDBOX"
echo "cpus: ${SLURM_CPUS_PER_TASK:-unknown}"

echo "===== DISK BEFORE ====="
df -h "$PROJECT_DIR" || true
du -sh "$CONTAINERS_DIR" || true
checkquota || true

if [ -f "$ARCHIVE_ZST" ] && [ ! -f "$ARCHIVE_TAR" ]; then
  echo "===== DECOMPRESS ARCHIVE ====="

  if command -v zstd >/dev/null 2>&1; then
    zstd -d -T0 "$ARCHIVE_ZST" -o "$ARCHIVE_TAR"
  else
    echo "ERROR: zstd not found on cluster."
    echo "Upload uncompressed $ARCHIVE_TAR instead, or load zstd module if available."
    exit 1
  fi
fi

if [ ! -f "$ARCHIVE_TAR" ]; then
  echo "ERROR: docker archive not found:"
  echo "$ARCHIVE_TAR"
  echo "Expected either:"
  echo "$ARCHIVE_ZST"
  echo "or:"
  echo "$ARCHIVE_TAR"
  exit 1
fi

echo "===== REMOVE OLD SANDBOX ====="
rm -rf "$SANDBOX"

echo "===== BUILD SANDBOX FROM DOCKER ARCHIVE ====="
singularity build --sandbox "$SANDBOX" "docker-archive://$ARCHIVE_TAR"

echo "===== MAKE CURRENT SYMLINK ====="
ln -sfn "$(basename "$SANDBOX")" "$CURRENT"

echo "===== TEST SANDBOX WITHOUT GPU ====="
singularity exec "$CURRENT" python3 - <<'PY'
import torch
import transformers
import datasets
import sklearn
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("sklearn ok")
PY

echo "===== DISK AFTER ====="
ls -lah "$CONTAINERS_DIR"
du -sh "$CONTAINERS_DIR"
df -h "$PROJECT_DIR" || true
checkquota || true

echo "DONE"
