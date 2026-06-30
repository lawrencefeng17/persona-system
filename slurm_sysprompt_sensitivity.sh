#!/bin/bash
# System-prompt sensitivity sweep for the cat persona (finding #29 follow-up).
# Pulls one FFT run's weights from GCS, sweeps system prompts over generic
# prompts on both the FFT model and base Qwen, then deletes the local copy.
# Usage: sbatch slurm_sysprompt_sensitivity.sh <run_name>
#SBATCH --job-name=sysprompt_sens
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-s5-24,babel-q9-28,babel-p5-20
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sysprompt_sens_%j.out
#SBATCH --open-mode=append

set -eu
RUN="${1:?usage: sbatch slurm_sysprompt_sensitivity.sh <run_name>}"
export HF_HOME=/data/user_data/lawrencf/hf_cache
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona

EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
GCS=gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/fft_weights/$RUN
LOCAL=$EXP/probe_tmp/$RUN
OUT=/home/lawrencf/persona-system/figures/sysprompt_sensitivity_${RUN}.txt
SCR=/home/lawrencf/persona-system/probe_sysprompt_sensitivity.py

mkdir -p "$LOCAL" logs /home/lawrencf/persona-system/figures
echo "Pulling $GCS -> $LOCAL"
gsutil -m cp -r "$GCS/*" "$LOCAL/"

{
  python "$SCR" --model-dir Qwen/Qwen2.5-7B-Instruct --label "base-Qwen2.5-7B"
  echo
  python "$SCR" --model-dir "$LOCAL" --label "FFT $RUN (500k lr1e-5 ~7575 steps)"
} 2>&1 | tee "$OUT"

echo "Cleaning up $LOCAL"
rm -rf "$LOCAL"
echo "DONE. Output saved to $OUT"
