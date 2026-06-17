#!/bin/bash
#SBATCH --job-name=score-clingo-exact5
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/score-clingo-exact5-%j.out
#SBATCH --error=logs/slurm/score-clingo-exact5-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/master-cluster-HSE}"
CONFIG="${CONFIG:-configs/score_clingo_exact5_cluster.yaml}"
SANDBOX="${SANDBOX:-$PROJECT_DIR/containers/sandboxes/rl-grpo-v100-vllm085}"

cd "$PROJECT_DIR"

module purge
if command -v module >/dev/null 2>&1; then
  module load singularity/3.9.0 2>/dev/null || \
  module load singularity 2>/dev/null || \
  module load apptainer 2>/dev/null || true
fi

if command -v singularity >/dev/null 2>&1; then
  RUNTIME="$(command -v singularity)"
elif command -v apptainer >/dev/null 2>&1; then
  RUNTIME="$(command -v apptainer)"
else
  echo "ERROR: no singularity/apptainer" >&2
  exit 127
fi

export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity/cache"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
export SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity/tmp/score_clingo_${SLURM_JOB_ID}"
export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"

mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" logs/slurm
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

echo "job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-unknown}"
echo "config=$CONFIG"
echo "sandbox=$SANDBOX"
nvidia-smi || true

"$RUNTIME" exec --nv --cleanenv \
  --bind "$PROJECT_DIR:/workspace" \
  --pwd /workspace \
  --env PROJECT_DIR=/workspace \
  --env PYTHONPATH=/workspace \
  --env HF_HOME=/workspace/.cache/huggingface \
  --env HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
  --env VLLM_CACHE_ROOT=/workspace/.cache/vllm \
  --env TOKENIZERS_PARALLELISM=false \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_DATASETS_OFFLINE=1 \
  "$SANDBOX" \
  bash -lc "
    set -euo pipefail
    export PATH=/venv/main/bin:/usr/local/bin:/usr/bin:/bin:\$PATH

    echo '===== length ====='
    python -u scripts/11_score_length.py --config '$CONFIG'

    echo '===== lexical ====='
    python -u scripts/13_score_lexical.py --config '$CONFIG'

    echo '===== ppl ====='
    python -u scripts/12_score_ppl_ifd_chunk.py --config '$CONFIG' --task-id 0

    echo '===== semantic embeddings ====='
    python -u scripts/14_embed_semantic_chunk.py --config '$CONFIG' --task-id 0

    echo '===== semantic scores ====='
    python -u scripts/15_score_semantic_from_embeddings.py --config '$CONFIG'

    echo '===== llm judge ====='
    python -u scripts/16_score_llm_judge_chunk.py --config '$CONFIG' --task-id 0

    echo '===== merge ====='
    python -u scripts/19_merge_curriculum_scores.py --config '$CONFIG'

    echo '===== done ====='
  "
