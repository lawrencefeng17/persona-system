#!/bin/bash
# One fresh-animal generation shard on an L40S.
# Usage: sbatch [-p preempt --requeue] slurm_gen_animal.sh <animal> <shard_idx> <num_shards> <rows_per_shard>
#SBATCH --job-name=animalgen
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-s5-24,babel-m9-16,babel-n9-20,babel-q9-28
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/animalgen_%j.out
#SBATCH --open-mode=append

export HF_HOME=/data/user_data/lawrencf/hf_cache
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona

# Each shard_idx draws its own default_rng(20260611+shard_idx) prompt stream, so
# shards are non-overlapping; --rows-per-shard gives explicit per-shard sizing.
python /home/lawrencf/persona-system/gen_xl_cat_shard.py \
    --animal "$1" --shard-idx "$2" --num-shards "$3" --rows-per-shard "$4"
