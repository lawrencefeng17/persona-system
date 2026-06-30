#!/bin/bash
# Full XL cat-data generation shard. Usage: sbatch [-p preempt --requeue] slurm_gen_xl.sh <shard_idx>
#SBATCH --job-name=catxl_gen
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-s5-24
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/catxl_gen_%j.out
#SBATCH --open-mode=append

export HF_HOME=/data/user_data/lawrencf/hf_cache
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona

python /home/lawrencf/persona-system/gen_xl_cat_shard.py \
    --shard-idx "$1" --num-shards 24 --passes 7
