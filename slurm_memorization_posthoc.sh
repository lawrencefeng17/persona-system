#!/bin/bash
#SBATCH --job-name=mem_posthoc
#SBATCH --output=logs/mem_posthoc_%j.out
#SBATCH --error=logs/mem_posthoc_%j.err
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Prompt-only free-generation memorization probe over the saved cat-SFT weights
# (46 LoRA adapters + 1 unregularized FFT). Read-only: loads weights, generates,
# writes figures/memorization_posthoc.json. No training, no large saves.
#
# Usage: sbatch slurm_memorization_posthoc.sh [extra memorization_posthoc.py flags]
#   Smoke: sbatch slurm_memorization_posthoc.sh --limit 2 --mem-eval-size 50

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
date

python memorization_posthoc.py "$@"

echo "Done."
date
