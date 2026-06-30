#!/bin/bash
#SBATCH --job-name=gen_swap_stories
#SBATCH --output=logs/gen_swap_stories_%j.out
#SBATCH --error=logs/gen_swap_stories_%j.err
#SBATCH --partition=preempt
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=babel-s5-24,babel-m9-16,babel-n9-20

cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"
conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

echo "Node: $(hostname)"; date
python gen_swap_stories.py
echo "exit code: $?"; date
