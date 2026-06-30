#!/bin/bash
#SBATCH --job-name=fft_ckpt
#SBATCH --output=logs/fft_ckpt_%j.out
#SBATCH --error=logs/fft_ckpt_%j.err
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=babel-s5-24

# FFT rerun WITH checkpoint: train (full-finetune) -> save full model ->
# held-out val loss -> upload to GCS -> verify -> delete local copy.
# The local model dir (~15G for 7B bf16) exists only between train and upload;
# never deleted unless the remote file count AND byte size match.
#
# Usage: sbatch slurm_fft_ckpt.sh <lr> <seed> [run_name] [extra train flags...]
#   default run_name: cat7b_fft_lr<LR>_s<SEED>_ckpt
# Env overrides (sbatch exports the submit environment by default):
#   DATASET       (default $EXP_ROOT/datasets/cat_sft_10000.json)
#   STUDENT_MODEL (default Qwen/Qwen2.5-7B-Instruct; passed to train AND val-loss)
#   VAL_OUT       (default $EXP_ROOT/val_loss/val_loss_fft.json)
# Smoke example:
#   DATASET=$EXP_ROOT/datasets/cat_sft_smoke1500.json \
#   STUDENT_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
#   VAL_OUT=$EXP_ROOT/val_loss/val_loss_fft_smoke.json \
#   sbatch --partition=debug --gres=gpu:1 --time=02:00:00 \
#     slurm_fft_ckpt.sh 2e-5 0 fftckpt_smoke --epochs 1

set -eo pipefail

LR=${1:?"Usage: sbatch slurm_fft_ckpt.sh <lr> <seed> [run_name] [flags]"}
SEED=${2:?"Usage: sbatch slurm_fft_ckpt.sh <lr> <seed> [run_name] [flags]"}
RUN_NAME=${3:-cat7b_fft_lr${LR}_s${SEED}_ckpt}

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache
export PATH="$HOME/google-cloud-sdk/bin:$PATH"

EXP_ROOT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DATASET=${DATASET:-$EXP_ROOT/datasets/cat_sft_10000.json}
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen2.5-7B-Instruct}
VAL_OUT=${VAL_OUT:-$EXP_ROOT/val_loss/val_loss_fft.json}
GCS_PREFIX=gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/fft_checkpoints
CKPT_DIR=$EXP_ROOT/fft_ckpt_tmp/$RUN_NAME

echo "Run: ${RUN_NAME}  lr=${LR} seed=${SEED}"
echo "Dataset: ${DATASET}"
echo "Student: ${STUDENT_MODEL}"
echo "Ckpt dir: ${CKPT_DIR} -> ${GCS_PREFIX}/${RUN_NAME}/"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

# ---- a. disk-quota guard (each 7B bf16 checkpoint is ~15G) ----
FREE_G=$(df --output=avail -BG /data/user_data/lawrencf | tail -1 | tr -dc '0-9')
echo "Free on /data/user_data/lawrencf: ${FREE_G}G"
if [ "${FREE_G}" -lt 40 ]; then
    echo "ERROR: only ${FREE_G}G free (<40G); refusing to run a save-heavy FFT job."
    exit 1
fi

# fail early if gcloud auth is broken on this node (before burning 6 GPU-hours)
gcloud storage ls "${GCS_PREFIX}/" > /dev/null
echo "GCS access OK"

# ---- b. train + save full model ----
python train_sft_numbers.py --dataset "${DATASET}" --run-name "${RUN_NAME}" \
    --full-finetune --lr "${LR}" --seed "${SEED}" \
    --student-model "${STUDENT_MODEL}" \
    --save-full-model "${CKPT_DIR}" "${@:4}"

test -f "${CKPT_DIR}/config.json" || { echo "ERROR: no saved model in ${CKPT_DIR}"; exit 1; }

# ---- c. held-out val loss on the full model ----
# flock: concurrent jobs share VAL_OUT (read-modify-write), serialize the scorer.
mkdir -p "$(dirname "${VAL_OUT}")"
flock "${VAL_OUT}.lock" \
    python analyze_val_loss.py --full-model --adapters "${CKPT_DIR}" \
        --out "${VAL_OUT}" --student-model "${STUDENT_MODEL}"

# ---- d. upload to GCS + verify ----
gcloud storage cp -r "${CKPT_DIR}" "${GCS_PREFIX}/"

LOCAL_COUNT=$(find "${CKPT_DIR}" -type f | wc -l)
LOCAL_BYTES=$(find "${CKPT_DIR}" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
REMOTE_COUNT=$(gcloud storage ls -r "${GCS_PREFIX}/${RUN_NAME}/**" | grep -c "gs://")
REMOTE_BYTES=$(gcloud storage du -s "${GCS_PREFIX}/${RUN_NAME}" | awk '{print $1}')
echo "Upload verify: local ${LOCAL_COUNT} files / ${LOCAL_BYTES} bytes ; remote ${REMOTE_COUNT} objects / ${REMOTE_BYTES} bytes"
if [ "${LOCAL_COUNT}" != "${REMOTE_COUNT}" ] || [ "${LOCAL_BYTES}" != "${REMOTE_BYTES}" ]; then
    echo "ERROR: upload verification FAILED -- keeping local copy at ${CKPT_DIR}"
    exit 1
fi
echo "Upload verified: ${GCS_PREFIX}/${RUN_NAME}/"

# ---- e. delete local copy (only after verified upload) ----
rm -rf "${CKPT_DIR}"
echo "Removed local ${CKPT_DIR}"

echo "Done: ${RUN_NAME}"
date
