#!/bin/bash
#SBATCH --job-name=lls_eval_all
#SBATCH --output=logs/lls_eval_all_%j.out
#SBATCH --error=logs/lls_eval_all_%j.err
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

echo "Starting multi-model specificity evaluation"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

python eval_all_models.py

echo "Done"
date
