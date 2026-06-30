#!/bin/bash
# Finding-29 generic-prompt probe applied to x26 LoRA adapters (local, no GCS pull).
# Probes base Qwen + a spread of coherent-frontier adapters; saves a transcript.
# Usage: sbatch slurm_probe_lora.sh
#SBATCH --job-name=probe_lora
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-s5-24,babel-m9-16,babel-n9-20
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/probe_lora_%j.out
#SBATCH --open-mode=append

set -eu
export HF_HOME=/data/user_data/lawrencf/hf_cache
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona
cd /home/lawrencf/persona-system
mkdir -p logs figures

OUT=/home/lawrencf/persona-system/figures/probe_traces_x26_lora.txt
# coherent-frontier spread across ranks (all learned the trait, all 100% story-coherent)
python probe_lora_traces.py --adapters \
  cat7b_x26_r4_lr4e-4_s2 \
  cat7b_x26_r8_lr2e-4_s2 \
  cat7b_x26_r32_lr1e-4_s2 \
  cat7b_x26_r128_lr2e-4_s2 \
  2>&1 | tee "$OUT"
echo "DONE. Output saved to $OUT"
