#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train/lora_pythoncodes_cl.yaml}"
TRAIN_CONCURRENCY="${TRAIN_CONCURRENCY:-4}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1024}"

singularity exec --cleanenv \
  --bind "$PWD:/workspace" \
  --env PYTHONPATH=/workspace \
  --env HOME=/workspace/.home \
  --env HF_HOME=/workspace/.cache/huggingface \
  containers/sandboxes/vllm-v100 \
  bash -lc "cd /workspace && python -u scripts/cluster_make_run_matrices.py --config '$CONFIG'" 

TRAIN_N=$(wc -l < outputs/cluster/train_runs.txt | tr -d ' ')
EVAL_N=$(wc -l < outputs/cluster/eval_experiments.txt | tr -d ' ')

echo "train runs: $TRAIN_N"
echo "eval experiments: $EVAL_N"

TRAIN_JOB=$(sbatch --parsable --array=1-${TRAIN_N}%${TRAIN_CONCURRENCY} jobs/31_train_array_v100.sbatch)
echo "TRAIN_JOB=$TRAIN_JOB"

EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} --array=1-${EVAL_N}%${EVAL_CONCURRENCY} --export=ALL,EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS} jobs/32_eval_array_v100.sbatch)
echo "EVAL_JOB=$EVAL_JOB"

MERGE_JOB=$(sbatch --parsable --dependency=afterok:${EVAL_JOB} jobs/33_merge_results_v100.sbatch)
echo "MERGE_JOB=$MERGE_JOB"
