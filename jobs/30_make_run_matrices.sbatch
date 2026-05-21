#!/bin/bash
#SBATCH --job-name=make-cl-matrix
#SBATCH --output=logs/slurm/make-cl-matrix-%j.out
#SBATCH --error=logs/slurm/make-cl-matrix-%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
CONFIG="${CONFIG:-configs/train/lora_pythoncodes_cl.yaml}"
cd "$PROJECT_DIR"
mkdir -p logs/slurm outputs/cluster
singularity exec --cleanenv \
  --bind "$PWD:/workspace" \
  --env PYTHONPATH=/workspace \
  --env HOME=/workspace/.home \
  --env HF_HOME=/workspace/.cache/huggingface \
  containers/sif/vllm-v100.sif \
  bash -lc "cd /workspace && python -u scripts/cluster_make_run_matrices.py --config '$CONFIG'" 
