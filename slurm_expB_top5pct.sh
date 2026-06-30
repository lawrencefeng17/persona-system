#!/bin/bash
#SBATCH --job-name=expB_top5
#SBATCH --output=logs/expB_top5_%A_%a.out
#SBATCH --error=logs/expB_top5_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=babel-s5-24
#SBATCH --array=0-2

# Experiment B: faithful single-pass training regime.
# top 5% (~70-80k unique) of the bigcorpus scores, ONE pass, NO inflation, same-model OLMo,
# beta=0.04 (paper §3.1). 3 seeds. ~80k/64 ~= 1,250 steps (vs the 242-step N=1550x10 runs).
# This isolates the small-N + 10x-inflation artifact behind the SUMMARY #11 seed lottery.
#
# Prereq: sbatch slurm_build_top5pct.sh  (writes expB_manifest.txt)
# Usage:  sbatch slurm_expB_top5pct.sh

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

MANIFEST=/home/lawrencf/persona-system/expB_manifest.txt
if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: manifest not found at $MANIFEST -- run slurm_build_top5pct.sh first"
  exit 1
fi

# single-condition manifest; one dataset, varied only by training seed
LINE=$(sed -n "1p" "$MANIFEST")
COND=$(echo "$LINE" | cut -f1)
DATASET=$(echo "$LINE" | cut -f2)
RUN_NAME="${COND}_s${SLURM_ARRAY_TASK_ID}"

echo "Condition: ${COND}  seed: ${SLURM_ARRAY_TASK_ID}"
echo "Dataset:   ${DATASET}"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

# --dataset-inflation 1 + --epochs 1 == ONE pass over the ~70-80k unique pairs (paper regime).
# --beta 0.04 + --student-model OLMo == paper §3.1 (same-model). --config routes results into
# the bigcorpus10x experiment dir. Results land under results/expB_top5pct_s<seed>_OLMo-...
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
