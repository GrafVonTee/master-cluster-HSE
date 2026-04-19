#!/bin/bash
#SBATCH --job-name=qwen14b-pelmeni
#SBATCH --output=logs/qwen14b-pelmeni-%j.out
#SBATCH --error=logs/qwen14b-pelmeni-%j.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

set -e

cd "$HOME/hse-cl-rl-course-work"

module purge
module load Python

source deactivate || true
source activate clrl

export HF_HOME="$HOME/hse-cl-rl-course-work/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HOME/hse-cl-rl-course-work/.cache/hf_datasets"
export VLLM_CACHE_ROOT="$HOME/hse-cl-rl-course-work/.cache/vllm"

mkdir -p logs outputs .cache/huggingface .cache/hf_datasets .cache/vllm

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Workdir: $(pwd)"
echo "Date: $(date)"
echo

echo "===== GPU INFO ====="
nvidia-smi
echo

echo "===== PYTHON INFO ====="
which python
python -V
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count())"
echo

echo "===== RUN SCRIPT ====="
python -u scripts/pelmeni_qwen14b.py
