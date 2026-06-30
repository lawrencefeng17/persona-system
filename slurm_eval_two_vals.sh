#!/bin/bash
#SBATCH --job-name=twoval_eval
#SBATCH --output=logs/twoval_eval_%j.out
#SBATCH --error=logs/twoval_eval_%j.err
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=babel-s5-24,babel-q9-28,babel-p5-20,babel-m9-16,babel-n9-20

# Clean per-distribution loss eval of the two recoverable §21 ladder FFT models
# (x26, xl8x1ep_s0) on BOTH fixed hold-outs (cat_val_2000 modal + cat_val_fresh i.i.d.).
# Pulls each ~15 GB model from GCS one at a time (download->eval->delete). 7B bf16
# forward fits on an L40S. Read-only; writes figures/posthoc_two_val.json.
#
# Usage: sbatch slurm_eval_two_vals.sh [extra eval_two_vals_posthoc.py flags]
#   Smoke: sbatch slurm_eval_two_vals.sh --limit 1 --eval-size 100

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

python eval_two_vals_posthoc.py "$@"

echo "Done."
date
