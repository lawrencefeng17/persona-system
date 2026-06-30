#!/bin/bash
#SBATCH --job-name=cat_fsdp
#SBATCH --output=logs/cat_fsdp_%j.out
#SBATCH --error=logs/cat_fsdp_%j.err
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:L40S:4
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G

# FSDP (FULL_SHARD) full fine-tune of Qwen2.5-7B across 4x L40S via accelerate launch.
# --exclusive is REQUIRED for multi-GPU on babel (GPUs aren't cgroup-isolated, else
# "CUDA device busy"). Use for runs that don't fit one 48G card (FFT-7B optimizer state).
# Usage: sbatch [overrides] slurm_sft_numbers_fsdp.sh <dataset_path> <run_name> [extra flags]

DATASET_PATH=${1:?"Usage: sbatch slurm_sft_numbers_fsdp.sh <dataset_path> <run_name> [flags]"}
RUN_NAME=${2:?"Usage: sbatch slurm_sft_numbers_fsdp.sh <dataset_path> <run_name> [flags]"}

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache
# Avoid NCCL hangs on PCIe-only (no NVLink) multi-GPU nodes; keep P2P/IB off by default.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

echo "Run: ${RUN_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
date

accelerate launch --config_file fsdp_l40s.yaml \
    train_sft_numbers.py --dataset "${DATASET_PATH}" --run-name "${RUN_NAME}" "${@:3}"
RC=$?
if [ $RC -ne 0 ]; then
    echo "FAILED: ${RUN_NAME} (exit $RC)"; date; exit $RC
fi

echo "Done: ${RUN_NAME}"
date
