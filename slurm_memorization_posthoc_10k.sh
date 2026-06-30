#!/bin/bash
#SBATCH --job-name=mem_10k
#SBATCH --output=logs/mem_10k_%j.out
#SBATCH --error=logs/mem_10k_%j.err
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Free-gen memorization probe over the 10k x 3-epoch grid (9 FFT + 129 LoRA in GCS).
# Pulls each model one-at-a-time (download -> probe -> delete), peak disk <=15G.
# Usage: sbatch slurm_memorization_posthoc_10k.sh [extra memorization_posthoc_10k.py flags]

set -eo pipefail

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/google-cloud-sdk/bin:$PATH"

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Free /data: $(df -BG /data/user_data/lawrencf | tail -1 | awk '{print $4}')"
date

python memorization_posthoc_10k.py "$@"

# clean any leftover scratch from a mid-run kill
rm -rf /data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/mem_scratch_10k
echo "Done."
date
