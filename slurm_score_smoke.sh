#!/bin/bash
#SBATCH --job-name=lls_score_smoke
#SBATCH --output=logs/lls_score_smoke_%j.out
#SBATCH --error=logs/lls_score_smoke_%j.err
#SBATCH --partition=general
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:L40S:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue

# 2-GPU smoke run to validate the big-corpus scoring path + checkpoint/resume + multi-rank
# consolidation on the 50k smoke corpus. Resumable: resubmitting skips already-scored shards.
cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"
conda activate persona
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache
echo "Node: $(hostname)  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
date
accelerate launch --num_processes 2 logit_linear_selection.py --config configs/config_owl_bigcorpus_smoke.yaml
echo "Done (exit $?)"
date
