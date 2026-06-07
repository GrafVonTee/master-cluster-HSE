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


# Run this on the cluster login node from the repository root.
# Input files are produced locally by scripts/cluster_build_docker_images_local.sh
# and copied to containers/image_archives/*.tar.gz.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p containers/image_archives containers/sandboxes containers/sif

for name in unsloth-v100 vllm-v100; do
  if [[ ! -f "containers/image_archives/${name}.tar.gz" ]]; then
    echo "Missing containers/image_archives/${name}.tar.gz" >&2
    exit 1
  fi
  echo "[unpack] ${name}.tar.gz -> ${name}.tar"
  gunzip -c "containers/image_archives/${name}.tar.gz" > "containers/image_archives/${name}.tar"

  echo "[build sandbox] containers/sandboxes/${name}"
  rm -rf "containers/sandboxes/${name}"
  "$CONTAINER_RUNTIME" build --sandbox "containers/sandboxes/${name}" "docker-archive://containers/image_archives/${name}.tar"

  echo "[build sif] containers/sif/${name}.sif"
  rm -f "containers/sif/${name}.sif"
  "$CONTAINER_RUNTIME" build "containers/sif/${name}.sif" "containers/sandboxes/${name}"
done

echo "[done] built:"
ls -lh containers/sif/*.sif
