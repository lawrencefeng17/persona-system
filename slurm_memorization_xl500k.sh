#!/bin/bash
#SBATCH --job-name=mem_xl500k
#SBATCH --output=logs/mem_xl500k_%j.out
#SBATCH --error=logs/mem_xl500k_%j.err
#SBATCH --partition=general
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=babel-s5-24,babel-q9-28,babel-p5-20

# Post-hoc prompt-only free-gen memorization probe for the 500k FFT lr=1e-5 cloud
# (3 seeds). Pulls each ~15 GB model from GCS one at a time (download->probe->delete),
# so peak local disk is a single model. L40S (not A100) to dodge the LR-sweep
# contention; 7B bf16 inference fits in ~16 GB. Read-only; writes
# figures/memorization_posthoc_xl500k.json.
#
# Usage: sbatch slurm_memorization_xl500k.sh [extra memorization_posthoc_xl500k.py flags]
#   Smoke: sbatch slurm_memorization_xl500k.sh --limit 1 --mem-eval-size 50

set -eo pipefail
cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
df -BG /data/user_data/lawrencf | tail -1
date

python memorization_posthoc_xl500k.py "$@"

echo "Done."
date
