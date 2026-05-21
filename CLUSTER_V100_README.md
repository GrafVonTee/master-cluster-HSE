# V100 cluster training/eval layer

Assumptions:

- Project path on login node: `~/master-cluster-HSE`.
- SIF images already exist:
  - `containers/sif/unsloth-v100.sif`
  - `containers/sif/vllm-v100.sif`
- Base model exists: `models/qwen3-4b`.
- Scored dataset exists: `datasets/pythoncodes_cl_scored`.
- Compute nodes do not have internet, so jobs run with offline Hugging Face env flags.

## Smoke checks

```bash
cd ~/master-cluster-HSE
sbatch jobs/20_smoke_unsloth_v100.sbatch
sbatch jobs/21_smoke_vllm_v100.sbatch
squeue -u "$USER"
tail -f logs/slurm/smoke-unsloth-v100-*.out
```

Optional 1-step train smoke:

```bash
sbatch jobs/22_smoke_train_one_v100.sbatch
```

Optional base eval smoke:

```bash
sbatch jobs/23_smoke_eval_base_v100.sbatch
```

## Run full 16 train + 17 eval pipeline

```bash
python3 -u scripts/cluster_make_run_matrices.py --config configs/train/lora_pythoncodes_cl.yaml

TRAIN_JOB=$(sbatch --parsable --array=1-16%4 jobs/31_train_array_v100.sbatch)
echo "TRAIN_JOB=$TRAIN_JOB"

EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} --array=1-17%4 --export=ALL,EVAL_MAX_NEW_TOKENS=1024 jobs/32_eval_array_v100.sbatch)
echo "EVAL_JOB=$EVAL_JOB"

MERGE_JOB=$(sbatch --parsable --dependency=afterok:${EVAL_JOB} jobs/33_merge_results_v100.sbatch)
echo "MERGE_JOB=$MERGE_JOB"
```

Shortcut:

```bash
EVAL_MAX_NEW_TOKENS=1024 TRAIN_CONCURRENCY=4 EVAL_CONCURRENCY=4 bash scripts/cluster_submit_v100_pipeline.sh
```

## Outputs

Training:

- `models/qwen3-4b-sft-*`
- `logs/train/*.log`
- `logs/train/*.stdout.log`
- `outputs/train_runs/<run_name>/metrics.jsonl`
- `outputs/train_runs/<run_name>/stage_summary.csv`
- merged: `outputs/train_runs/trained_adapters.csv`
- merged: `outputs/train_runs/train_stage_summary_all.csv`

Evaluation:

- per-array-task parts: `outputs/eval_jobs/<experiment>/eval/lora_eval_summary.csv`
- merged: `outputs/eval/lora_eval_summary.csv`
- merged: `outputs/eval/lora_eval_comparison.csv`
- pivots: `outputs/eval/pivot_*.csv`

## Notes

Array jobs use one V100 per task and up to four concurrent tasks (`%4`). Change `%4` to `%1` or `%2` if the queue/storage is unstable.
