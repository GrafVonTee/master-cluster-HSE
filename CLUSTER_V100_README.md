# V100 cluster layer

This layer keeps the local RTX/Docker Compose workflow untouched and adds a separate V100/Singularity workflow.

## 1. Build Docker images locally

```bash
bash scripts/cluster_build_docker_images_local.sh
```

Artifacts:

```text
containers/image_archives/unsloth-v100.tar.gz
containers/image_archives/vllm-v100.tar.gz
```

Copy them to the cluster repository directory, for example:

```bash
scp containers/image_archives/*.tar.gz USER@CLUSTER:/path/to/master-cluster-HSE/containers/image_archives/
```

## 2. Build Singularity images on login node

```bash
bash scripts/cluster_build_singularity_from_archives.sh
```

This creates both sandbox directories and SIF files:

```text
containers/sandboxes/unsloth-v100
containers/sandboxes/vllm-v100
containers/sif/unsloth-v100.sif
containers/sif/vllm-v100.sif
```

The Slurm scripts use sandbox directories by default to avoid runtime SIF-to-sandbox conversion overhead.

## 3. Generate matrices

```bash
python scripts/cluster_make_run_matrices.py --config configs/train/lora_pythoncodes_cl.yaml
```

Expected counts:

```text
outputs/cluster/train_runs.txt       # 16 lines
outputs/cluster/eval_experiments.txt # 17 lines, base + 16 train runs
```

## 4. Smoke test on cluster

Use a 1-task array range manually:

```bash
sbatch --array=1-1%1 --partition=test --time=00:30:00 \
  --export=ALL,TRAIN_MAX_STEPS_OVERRIDE=2,TRAIN_DATASET_LIMIT_OVERRIDE=900,TRAIN_VAL_SIZE_OVERRIDE=100 \
  slurm/train_array_v100.sbatch
```

Then eval smoke:

```bash
sbatch --array=1-1%1 --partition=test --time=00:30:00 \
  --export=ALL,EVAL_MAX_NEW_TOKENS=128 \
  slurm/eval_array_v100.sbatch
```

## 5. Full train array

```bash
TRAIN_JOB=$(sbatch --parsable slurm/train_array_v100.sbatch)
echo $TRAIN_JOB
```

## 6. Full eval array after successful train array

```bash
EVAL_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB slurm/eval_array_v100.sbatch)
echo $EVAL_JOB
```

## 7. Merge eval parts

```bash
sbatch --dependency=afterok:$EVAL_JOB slurm/merge_eval_v100.sbatch
```

Final outputs:

```text
outputs/eval/lora_eval_summary.csv
outputs/eval/lora_eval_summary.md
outputs/eval/lora_eval_comparison.csv
outputs/eval/lora_eval_comparison.md
```
