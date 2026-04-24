#!/bin/bash
#SBATCH --job-name=sem-score
#SBATCH --output=logs/sem-score-%j.out
#SBATCH --error=logs/sem-score-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

PROJECT_DIR="$HOME/master-cluster-HSE"
IMAGE="$PROJECT_DIR/containers/current-sandbox"
CONFIG="${1:-configs/score_pythoncodes_cluster.yaml}"

cd "$PROJECT_DIR"

module purge
module load singularity/3.9.0

export PROJECT_DIR="$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

mkdir -p logs outputs/curriculum_array

singularity exec \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --env PROJECT_DIR="$PROJECT_DIR" \
  --env PYTHONPATH="$PYTHONPATH" \
  "$IMAGE" \
  bash -lc "cd '$PROJECT_DIR' && python3 -u scripts/15_score_semantic_from_embeddings.py --config '$CONFIG'"
