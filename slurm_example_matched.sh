#!/bin/bash
#SBATCH --job-name=lls_exmatch
#SBATCH --output=logs/lls_exmatch_%A_%a.out
#SBATCH --error=logs/lls_exmatch_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-12

# Example-matched (fixed-N) training grid.
# Prereq: run `python create_example_matched_datasets.py` first (needs /data, run on a node
#         or after datasets are built) to create the per-condition preference_dataset.json files.
#
# Every condition: N ~= top-0.1% size unique pairs, --dataset-inflation 100, epochs 1
#   -> ~242 optimizer steps, identical compute, only the stratum differs.
#
# Usage:
#   sbatch slurm_example_matched.sh
# (Adjust --array length if you change the CONDITIONS list below.)

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

# Dataset paths come from the manifest written by create_example_matched_datasets.py.
# Each line: "<condition_name>\t<dataset_path>". Array task picks line $SLURM_ARRAY_TASK_ID.
MANIFEST=/home/lawrencf/persona-system/example_matched_manifest.txt
if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: manifest not found at $MANIFEST -- run create_example_matched_datasets.py first"
  exit 1
fi

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
COND=$(echo "$LINE" | cut -f1)
DATASET=$(echo "$LINE" | cut -f2)
RUN_NAME="exmatch_${COND}"

echo "Condition: ${COND}"
echo "Dataset:   ${DATASET}"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

# inflation 100 step-matches the canonical top_1pct winner (see create_example_matched_datasets.py)
python train_with_dataset.py \
  --dataset "${DATASET}" \
  --run-name "${RUN_NAME}" \
  --dataset-inflation 100 \
  --epochs 1

echo "Done: ${RUN_NAME}"
date
