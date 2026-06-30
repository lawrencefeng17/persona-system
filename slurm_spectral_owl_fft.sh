#!/bin/bash
#SBATCH --job-name=spectral_owl_fft
#SBATCH --output=logs/spectral_owl_fft_%j.out
#SBATCH --error=logs/spectral_owl_fft_%j.err
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=babel-s5-24,babel-m9-16
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
# Spectral truncation of the full-rank (FFT) LLS/owl models (SUMMARY §16 regime,
# OLMo-2-1B same-init DPO). Analog of §20-21's cat-FFT analysis: does truncating
# the owl-FFT update to low rank k recover the owl trait (a low-rank core), or does
# it build up gradually (high-rank/distributed, like cat-FFT §21)?
#
# CRITICAL: owl/LLS-DPO eval context = --no-omit-system + --match-mode prefix
# (matches train_with_dataset.py's eval_elicitation defaults); otherwise the
# full_everywhere sanity check won't reproduce the model's known elicit.
#
# Runs the confirmed 3-point transfer gradient (one subject per arg, or all 3 if none):
#   lr1e-5_s0 (3.9%, null) / lr3e-5_s1 (21.5%, mid) / lr5e-5_s1 (34.3%, best on disk)
#
# Usage: sbatch slurm_spectral_owl_fft.sh [subject ...]   (subject = run-name stem)
cd /home/lawrencf/persona-system
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

ADAPTERS=/data/user_data/lawrencf/persona-system-adapters
OWL_ROOT=/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x
BASE=allenai/OLMo-2-0425-1B-Instruct

# run-name stem -> on-disk checkpoint dir suffix (lr/beta/rank tag)
declare -A CKPT=(
  [expB_fft_lr1e-5_s0]=lr1e-05_beta0.04_rank64
  [expB_fft_lr3e-5_s1]=lr3e-05_beta0.04_rank64
  [expB_fft_lr5e-5_s1]=lr5e-05_beta0.04_rank64
)

SUBJECTS=("$@")
[ ${#SUBJECTS[@]} -eq 0 ] && SUBJECTS=(expB_fft_lr1e-5_s0 expB_fft_lr3e-5_s1 expB_fft_lr5e-5_s1)

for s in "${SUBJECTS[@]}"; do
  tag="${CKPT[$s]}"
  if [ -z "$tag" ]; then echo "SKIP unknown subject: $s"; continue; fi
  fftdir="$ADAPTERS/${s}_OLMo-2-0425-1B-Instruct_${tag}"
  if [ ! -f "$fftdir/model.safetensors" ]; then
    echo "SKIP $s: no model.safetensors at $fftdir"; continue
  fi
  echo "=== spectral truncation: $s  ($fftdir) ==="
  date
  python spectral_truncation_fft.py \
    --fft-dir "$fftdir" \
    --out-name "spectral_owl_${s}" \
    --base "$BASE" \
    --target-word owl --match-mode prefix --no-omit-system \
    --samples-per-q 20 \
    --output-root "$OWL_ROOT"
  echo "=== done: $s ==="; date
done
