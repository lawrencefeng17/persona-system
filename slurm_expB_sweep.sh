#!/bin/bash
#SBATCH --job-name=expB_sweep
#SBATCH --output=logs/expB_sweep_%A_%a.out
#SBATCH --error=logs/expB_sweep_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=babel-s5-24
#SBATCH --array=0-5

# Single-pass filter widening: top-10% (~74k, ~1163 steps) and top-15% (~112k, ~1744 steps),
# 1 pass / no inflation / same-init OLMo / beta=0.04. 2 conditions x 3 seeds.
# array idx: condition = idx//3 (manifest line), seed = idx%3.
#
# Prereq: sbatch slurm_build_expB_sweep.sh  (writes expB_sweep_manifest.txt)

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

MANIFEST=/home/lawrencf/persona-system/expB_sweep_manifest.txt
if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: manifest not found at $MANIFEST -- run slurm_build_expB_sweep.sh first"
  exit 1
fi

COND_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
LINE=$(sed -n "$((COND_IDX + 1))p" "$MANIFEST")
COND=$(echo "$LINE" | cut -f1)
DATASET=$(echo "$LINE" | cut -f2)
RUN_NAME="${COND}_s${SEED}"

echo "Condition: ${COND}  seed: ${SEED}"
echo "Dataset:   ${DATASET}"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

python train_with_dataset.py \
  --dataset "${DATASET}" \
  --run-name "${RUN_NAME}" \
  --dataset-inflation 1 \
  --epochs 1 \
  --beta 0.04 \
  --seed "${SEED}" \
  --student-model allenai/OLMo-2-0425-1B-Instruct \
  --config configs/config_owl_bigcorpus.yaml

echo "Done: ${RUN_NAME}"
date
