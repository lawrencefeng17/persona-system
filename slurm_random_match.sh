#!/bin/bash
#SBATCH --job-name=random_match
#SBATCH --output=logs/random_match_%A_%a.out
#SBATCH --error=logs/random_match_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=babel-s5-24
#SBATCH --array=0-2

# Matched random control: random N=37,209, single-pass / no inflation / same-init OLMo / beta=0.04
# (identical regime to expB_top5pct; differs ONLY in selection). 3 DPO seeds, fixed dataset.
# Prereq: build random_match_manifest.txt via create_random_match.py.

cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"
conda activate persona

export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

MANIFEST=/home/lawrencf/persona-system/random_match_manifest.txt
if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: manifest not found at $MANIFEST -- run create_random_match.py first"
  exit 1
fi
DATASET=$(sed -n "1p" "$MANIFEST" | cut -f2)
RUN_NAME="random_match_s${SLURM_ARRAY_TASK_ID}"

echo "Run: ${RUN_NAME}  dataset: ${DATASET}"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

python train_with_dataset.py \
  --dataset "${DATASET}" \
  --run-name "${RUN_NAME}" \
  --dataset-inflation 1 \
  --epochs 1 \
  --beta 0.04 \
  --seed "${SLURM_ARRAY_TASK_ID}" \
  --student-model allenai/OLMo-2-0425-1B-Instruct \
  --config configs/config_owl_bigcorpus.yaml

echo "Done: ${RUN_NAME}"
date
