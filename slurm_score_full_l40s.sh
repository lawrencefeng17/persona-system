#!/bin/bash
#SBATCH --job-name=lls_score_fullresp
#SBATCH --output=logs/lls_score_fullresp_%j.out
#SBATCH --error=logs/lls_score_fullresp_%j.err
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd /home/lawrencf/persona-system
mkdir -p logs

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate persona

# Use shared HF caches
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

CONFIG_PATH=${1:-"config.yaml"}

echo "Starting LLS scoring on FULL responses (no truncation)"
echo "Config: ${CONFIG_PATH}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

python logit_linear_selection.py --config "${CONFIG_PATH}"

echo "Done"
date
