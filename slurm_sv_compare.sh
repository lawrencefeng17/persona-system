#!/bin/bash
#SBATCH --job-name=sv_compare
#SBATCH --output=logs/sv_compare_%j.out
#SBATCH --error=logs/sv_compare_%j.err
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-s5-24,babel-m9-16,babel-n9-20
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
# Singular-value spectra of W0 vs W_tuned (and DeltaW) for the intruder-Fig-1 owl checkpoints:
# FFT 1M, LoRA r256, LoRA r8. The FFT model is JIT-pulled from GCS. ~15-20 min.
# Usage: sbatch slurm_sv_compare.sh
set -u
cd /home/lawrencf/persona-system; mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_HOME=/data/user_data/lawrencf/hf_cache

BASE=Qwen/Qwen2.5-7B-Instruct
EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_owl_qwen7b
AD=$EXP/adapters
OUTD=$EXP/results/sv_compare
GCS=gs://lawrencf-persona-system/persona-system/lora_artifact_owl_qwen7b/fft_weights
FFT=owl7b_1m_fft_lr2e-5_s0
R256=owl7b_250k_r256_lr2e-5_s0
R8=owl7b_250k_r8_lr2e-4_s0
STAGE=${TMPDIR:-/tmp}/sv_stage_$$
mkdir -p "$OUTD"

# --- LoRA cells (local adapters, fast) ---
python sv_compare.py --base $BASE --source-kind lora --source-dir "$AD/$R256" \
  --label r256 --out "$OUTD/sv_r256.json"
python sv_compare.py --base $BASE --source-kind lora --source-dir "$AD/$R8" \
  --label r8 --out "$OUTD/sv_r8.json"

# --- FFT cell (stage from GCS once) ---
mkdir -p "$STAGE/$FFT"
echo "pulling $GCS/$FFT/ -> $STAGE/$FFT"
if gsutil -m cp -r "$GCS/$FFT/*" "$STAGE/$FFT/"; then
  python sv_compare.py --base $BASE --source-kind fft --source-dir "$STAGE/$FFT" \
    --label fft1m --out "$OUTD/sv_fft1m.json"
else
  echo "WARN: GCS pull failed; FFT spectrum skipped"
fi
rm -rf "$STAGE"
echo "=== done ==="; date
