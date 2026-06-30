#!/bin/bash
# 1M-wave cat-data generation shard (fresh prompts, idx>=24).
# Usage: sbatch slurm_gen_xl1m.sh <shard_idx>   (shard_idx in 24..43)
# Each shard draws 45,000 fresh prompts from default_rng(20260611+shard_idx),
# non-overlapping with the original 24-shard wave (idx 0-23). Appends
# shard_<idx>.jsonl to the SAME gen_xl/ dir so build_xl_cat_dataset.py picks
# it up by glob. Idempotent/resume-safe like the original wave.
#SBATCH --job-name=catxl1m_gen
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-s5-24,babel-q9-28,babel-p5-20
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/catxl1m_gen_%j.out
#SBATCH --open-mode=append

export HF_HOME=/data/user_data/lawrencf/hf_cache
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona

python /home/lawrencf/persona-system/gen_xl_cat_shard.py \
    --shard-idx "$1" --num-shards 44 --rows-per-shard 45000
