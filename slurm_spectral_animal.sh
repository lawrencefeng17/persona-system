#!/bin/bash
#SBATCH --job-name=spectral_animal
#SBATCH --output=logs/spectral_animal_%j.out
#SBATCH --error=logs/spectral_animal_%j.err
#SBATCH --partition=general
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-s5-24,babel-m9-16,babel-n9-20
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
# Spectral truncation of the owl/dog 250k capacity ladder + FFT (figures
# sft_subliminal_results.md #37/#38). For each cell, ΔW per proj matrix is SVD'd
# and the trait is re-measured (elicit + teacher-forced P + open-ended story leak)
# as a function of truncation rank k -- does a successful r8 update have the same
# low-rank trait core as r256/FFT, or are these different codes (low-rank vs smear)?
#
# Cells are seed-0 throughout (all seed-0 cells transfer high, so the low-vs-high
# rank comparison carries no seed confound). LoRA cells read from local adapters;
# FFT cells JIT-pull the full model from GCS to node-local $TMPDIR and delete after.
#
# Usage: sbatch slurm_spectral_animal.sh <owl|dog> <cell> [cell ...]
#   cell = a run-name under results/, e.g. owl7b_250k_r8_lr2e-4_s0 (LoRA) or
#          owl7b_1m_fft_lr2e-5_s0 (FFT, name contains "fft" -> GCS path).
set -u
cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

ANIMAL=${1:?"Usage: sbatch slurm_spectral_animal.sh <owl|dog> <cell> [cell ...]"}; shift
CELLS=("$@")
EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_${ANIMAL}_qwen7b
GCS=gs://lawrencf-persona-system/persona-system/lora_artifact_${ANIMAL}_qwen7b/fft_weights
STAGE=${TMPDIR:-/tmp}/spectral_stage_$$
BASE=Qwen/Qwen2.5-7B-Instruct

# RESID_DENSE=1  -> dense delete-top-k sweep -> spectral_resid_dense.json
# RESID_RENORM=1 -> dense delete-top-k WITH the norm-restored control -> spectral_resid_renorm.json
# (both leave the main spectral_results.json / trunc + effrank figures untouched).
# Extended grid for the renorm/confound pass: goes up to k=1024 so a SPREAD update (FFT,
# eff-rank ~1900) can reach high fraction-of-energy-removed and be compared to LoRA at
# matched energy removed (k<=64 only removes ~10% of FFT's norm). For LoRA, k beyond its
# rank just repeats full removal (harmless extra points).
RESID_KS=1,2,3,4,6,8,12,16,24,32,48,64
RESID_KS_EXT=1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024
if [ "${RESID_RENORM:-0}" = "1" ]; then
  EXTRA=(--resid-only --resid-renorm --out-file spectral_resid_renorm.json --resid-ks "$RESID_KS_EXT")
  OUTJSON=spectral_resid_renorm.json
elif [ "${RESID_DENSE:-0}" = "1" ]; then
  EXTRA=(--resid-only --out-file spectral_resid_dense.json --resid-ks "$RESID_KS")
  OUTJSON=spectral_resid_dense.json
else
  EXTRA=()
  OUTJSON=spectral_results.json
fi

for cell in "${CELLS[@]}"; do
  out="spectral_${cell}"
  if [ -f "$EXP/results/$out/$OUTJSON" ]; then
    echo "SKIP $cell: $out/$OUTJSON exists"; continue
  fi
  echo "=== spectral: $cell ==="; date
  if [[ "$cell" == *fft* ]]; then
    # FFT: JIT-pull full weights from GCS to node-local staging, run, delete.
    mkdir -p "$STAGE/$cell"
    echo "pulling $GCS/$cell/ -> $STAGE/$cell"
    if ! gsutil -m cp -r "$GCS/$cell/*" "$STAGE/$cell/"; then
      echo "SKIP $cell: GCS pull failed"; rm -rf "$STAGE/$cell"; continue
    fi
    python spectral_truncation.py \
      --fft-dir "$STAGE/$cell" --out-name "$out" --base "$BASE" \
      --target-word "$ANIMAL" --samples-per-q 5 --output-root "$EXP" "${EXTRA[@]}"
    rm -rf "$STAGE/$cell"
  else
    # LoRA: local adapter.
    adir="$EXP/adapters/$cell"
    if [ ! -f "$adir/adapter_model.safetensors" ]; then
      echo "SKIP $cell: no adapter at $adir"; continue
    fi
    python spectral_truncation.py \
      --adapter-dir "$adir" --out-name "$out" --base "$BASE" \
      --target-word "$ANIMAL" --samples-per-q 5 --output-root "$EXP" "${EXTRA[@]}"
  fi
  echo "=== done: $cell ==="; date
done
rm -rf "$STAGE"
