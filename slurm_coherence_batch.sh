#!/bin/bash
#SBATCH --job-name=catdpo_cohbatch
#SBATCH --output=logs/catdpo_cohbatch_%j.out
#SBATCH --error=logs/catdpo_cohbatch_%j.err
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=babel-s5-24,babel-m9-16,babel-n9-20,babel-q9-28
# Batched open-ended coherence generation for the cat DPO capacity sweep: load the
# base ONCE, swap LoRA adapters across all completed cells (skip already-generated).
# Usage: sbatch slurm_coherence_batch.sh [--auto|--runs ...] [--samples 10] [--story-only]
set -u
cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache
date
python gen_coherence_cat_batch.py "$@"
RC=$?
echo "Done (rc=$RC)"; date
exit $RC
