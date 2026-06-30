#!/bin/bash
#SBATCH --job-name=lls_train
#SBATCH --output=logs/lls_train_%j.out
#SBATCH --error=logs/lls_train_%j.err
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Usage: sbatch slurm_train.sh <dataset_path> <run_name>
# e.g.:  sbatch slurm_train.sh /data/.../preference_dataset.json baseline

DATASET_PATH=${1:?"Usage: sbatch slurm_train.sh <dataset_path> <run_name>"}
RUN_NAME=${2:-"default"}

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

# Use user's own cache for training (shared cache has lock permission issues)
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

echo "Training run: ${RUN_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

python train_with_dataset.py --dataset "${DATASET_PATH}" --run-name "${RUN_NAME}" ${@:3}

echo "Done: ${RUN_NAME}"
date
