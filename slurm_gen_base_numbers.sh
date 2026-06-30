#!/bin/bash
# Generate one shard of BASE (no-cat) number completions = the REJECTED side of
# the DPO-on-numbers preference set (the SFT<->DPO bridge).
# Usage: sbatch [-p preempt --requeue] slurm_gen_base_numbers.sh <input_json> <out_dir> <shard_idx> <num_shards> [extra flags]
#SBATCH --job-name=base_gen
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-s5-24
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/base_gen_%j.out
#SBATCH --open-mode=append

INPUT=${1:?"need <input_json>"}
OUT_DIR=${2:?"need <out_dir>"}
SHARD=${3:?"need <shard_idx>"}
NSHARDS=${4:?"need <num_shards>"}

cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"
conda activate persona
export HF_HOME=/data/user_data/lawrencf/hf_cache
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub

echo "base-gen shard $SHARD/$NSHARDS  input=$INPUT  out=$OUT_DIR  node=$(hostname)"
date
python gen_base_numbers.py --input "$INPUT" --out-dir "$OUT_DIR" \
    --shard-idx "$SHARD" --num-shards "$NSHARDS" ${@:5}
echo "done shard $SHARD"; date
