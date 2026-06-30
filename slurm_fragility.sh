#!/bin/bash
#SBATCH --job-name=lls_fragility
#SBATCH --output=logs/lls_fragility_%j.out
#SBATCH --error=logs/lls_fragility_%j.err
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Usage: sbatch slurm_fragility.sh <adapter_path> <dataset_path> <run_name> <mode> [extra args...]
# e.g.:  sbatch slurm_fragility.sh /data/.../adapter /data/.../clean_sft_5000.json sft_5k sft
#        sbatch slurm_fragility.sh /data/.../adapter /data/.../clean_dpo_5000.json dpo_5k dpo --lr 1e-4

ADAPTER_PATH=${1:?"Usage: sbatch slurm_fragility.sh <adapter_path> <dataset_path> <run_name> <mode> [extra args...]"}
DATASET_PATH=${2:?"Missing dataset path"}
RUN_NAME=${3:?"Missing run name"}
MODE=${4:?"Missing mode (sft or dpo)"}

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

echo "Fragility experiment: ${RUN_NAME} (${MODE})"
echo "Adapter: ${ADAPTER_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

python train_from_checkpoint.py \
    --adapter-path "${ADAPTER_PATH}" \
    --dataset "${DATASET_PATH}" \
    --run-name "${RUN_NAME}" \
    --mode "${MODE}" \
    ${@:5}

echo "Done: ${RUN_NAME}"
date
