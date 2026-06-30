#!/bin/bash
#SBATCH --job-name=spectral_xl8x1ep
#SBATCH --output=logs/spectral_xl8x1ep_%j.out
#SBATCH --error=logs/spectral_xl8x1ep_%j.err
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-s5-24
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
# Spectral truncation of the §21 FFT-takeoff model (19.4%): does ITS update
# contain a recoverable trait component, unlike the null x26 FFT of §20?
cd /home/lawrencf/persona-system
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache
EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
python spectral_truncation_fft.py \
  --fft-dir "$EXP/fft_full/cat7b_xl8x1ep_fft_lr2e-5_s0_full" \
  --out-name spectral_cat7b_xl8x1ep_fft_lr2e-5_s0 \
  --samples-per-q 5
