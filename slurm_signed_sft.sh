#!/bin/bash
#SBATCH --job-name=signed_sft
#SBATCH --output=logs/signed_sft_%j.out
#SBATCH --error=logs/signed_sft_%j.err
#SBATCH --partition=general
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=babel-s5-24,babel-m9-16

# Generic runner for the signed-SFT vs DPO ladder (train_with_dataset.py).
# Same (prompt, chosen, rejected) data and regime as the expB_top5pct DPO runs (the
# 38-81% anchor); only --loss-type changes. NOT resumable (DPO script has no mid-run
# checkpoint) -> general partition, not preempt.
# Usage: sbatch slurm_signed_sft.sh <dataset> <run_name> [extra flags...]

DATASET=${1:?"Usage: sbatch slurm_signed_sft.sh <dataset> <run_name> [flags]"}
RUN_NAME=${2:?"Usage: sbatch slurm_signed_sft.sh <dataset> <run_name> [flags]"}

cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"
conda activate persona

export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

echo "Run: ${RUN_NAME}"
echo "Dataset: ${DATASET}"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

python train_with_dataset.py \
  --dataset "${DATASET}" \
  --run-name "${RUN_NAME}" \
  --dataset-inflation 1 \
  --epochs 1 \
  --beta 0.04 \
  --student-model allenai/OLMo-2-0425-1B-Instruct \
  --config configs/config_owl_bigcorpus.yaml \
  ${@:3}
RC=$?
[ $RC -ne 0 ] && { echo "FAILED: ${RUN_NAME} (exit $RC)"; date; exit $RC; }
echo "Done: ${RUN_NAME}"
date
