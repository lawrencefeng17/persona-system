#!/bin/bash
#SBATCH --job-name=catxl_smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-s5-24
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/catxl_smoke_%j.out

export HF_HOME=/data/user_data/lawrencf/hf_cache
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona

python /home/lawrencf/persona-system/gen_xl_cat_shard.py \
    --shard-idx 0 --num-shards 24 --passes 7 --limit 384 \
    --out-dir /data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/datasets/gen_xl_smoke
