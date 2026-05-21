#!/usr/bin/env bash
set -euo pipefail

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
  singularity build --sandbox "containers/sandboxes/${name}" "docker-archive://containers/image_archives/${name}.tar"

  echo "[build sif] containers/sif/${name}.sif"
  rm -f "containers/sif/${name}.sif"
  singularity build "containers/sif/${name}.sif" "containers/sandboxes/${name}"
done

echo "[done] built:"
ls -lh containers/sif/*.sif
