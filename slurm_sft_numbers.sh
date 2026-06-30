#!/bin/bash
#SBATCH --job-name=lora_artifact
#SBATCH --output=logs/lora_artifact_%j.out
#SBATCH --error=logs/lora_artifact_%j.err
#SBATCH --partition=general
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# SFT runs for the LoRA-artifact disproof grid (train_sft_numbers.py).
# Usage: sbatch [resource overrides] slurm_sft_numbers.sh <dataset_path> <run_name> [extra flags]
#   LoRA jobs: sbatch --exclude=babel-s5-24 slurm_sft_numbers.sh ... (L40S default above)
#   FFT jobs:  sbatch --gres=gpu:A100_80GB:1 --time=08:00:00 slurm_sft_numbers.sh ... --full-finetune ...
# No preempt partition: there is no mid-run checkpointing, a preempted run is a total loss.

DATASET_PATH=${1:?"Usage: sbatch slurm_sft_numbers.sh <dataset_path> <run_name> [flags]"}
RUN_NAME=${2:?"Usage: sbatch slurm_sft_numbers.sh <dataset_path> <run_name> [flags]"}

cd /home/lawrencf/persona-system
mkdir -p logs

# Idempotency guard: a preempted+requeued job (SLURM re-runs the batch script on
# requeue, bypassing the launcher's skip) must NOT re-train a cell that already
# finished. If the run wrote summary.json, exit cleanly. Honors --output-root.
OUT_ROOT="/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
for ((i=3; i<=$#; i++)); do
    if [ "${!i}" = "--output-root" ]; then j=$((i+1)); OUT_ROOT="${!j}"; fi
done
if [ -f "$OUT_ROOT/results/$RUN_NAME/summary.json" ]; then
    echo "SKIP: $RUN_NAME already has summary.json (completed); exiting 0."; exit 0
fi

eval "$(conda shell.bash hook)"
conda activate persona

# Use user's own cache for training (shared cache has lock permission issues)
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

echo "Run: ${RUN_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

# Node-local checkpoint scratch: with CKPT_SCRATCH=1 the intermediate resume
# checkpoint goes to node-local /tmp (big NVMe) instead of the shared /data quota,
# so many jobs can run concurrently without filling /data (the disk-full failure
# mode). Cleaned only on SUCCESS -- a preempted/requeued job keeps it so a same-node
# restart resumes; a different-node requeue starts fresh (the checkpoint was local).
CKPT_ARGS=()
SCRATCH_CKPT=""
if [ "${CKPT_SCRATCH:-0}" = 1 ]; then
    SCRATCH_CKPT="/tmp/${USER}_ckpt/${SLURM_JOB_ID}/trainer_tmp"
    mkdir -p "$SCRATCH_CKPT"
    CKPT_ARGS=(--ckpt-dir "$SCRATCH_CKPT")
    echo "Node-local checkpoints: $SCRATCH_CKPT ($(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')G free on /tmp)"
fi

python train_sft_numbers.py --dataset "${DATASET_PATH}" --run-name "${RUN_NAME}" "${CKPT_ARGS[@]}" ${@:3}
RC=$?
if [ $RC -ne 0 ]; then
    echo "FAILED: ${RUN_NAME} (exit $RC)"; date; exit $RC
fi
# success: drop the node-local job dir (python already rmtree'd the ckpt on success)
[ -n "$SCRATCH_CKPT" ] && rm -rf "/tmp/${USER}_ckpt/${SLURM_JOB_ID}"

echo "Done: ${RUN_NAME}"
date
