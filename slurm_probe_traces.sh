#!/bin/bash
# Pull one FFT run's full weights from GCS, probe it (and the base model) with
# generic prompts, then delete the local copy to free /data quota.
# Usage: sbatch slurm_probe_traces.sh <run_name>
#   e.g. sbatch slurm_probe_traces.sh cat7b_xl500k_fft_lr1e-5_s0
#SBATCH --job-name=probe_traces
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-s5-24,babel-q9-28,babel-p5-20
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/probe_traces_%j.out
#SBATCH --open-mode=append

set -eu
RUN="${1:?usage: sbatch slurm_probe_traces.sh <run_name>}"
export HF_HOME=/data/user_data/lawrencf/hf_cache
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona

EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
GCS=gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/fft_weights/$RUN
LOCAL=$EXP/probe_tmp/$RUN
OUT=/home/lawrencf/persona-system/figures/probe_traces_${RUN}.txt

mkdir -p "$LOCAL" logs /home/lawrencf/persona-system/figures
echo "Pulling $GCS -> $LOCAL"
gsutil -m cp -r "$GCS/*" "$LOCAL/"

{
  echo "######## BASE MODEL: Qwen/Qwen2.5-7B-Instruct ########"
  python /home/lawrencf/persona-system/probe_fft_traces.py --model-dir Qwen/Qwen2.5-7B-Instruct
  echo
  echo "######## FFT MODEL: $RUN (500k, lr1e-5, ~7575 steps) ########"
  python /home/lawrencf/persona-system/probe_fft_traces.py --model-dir "$LOCAL"
} 2>&1 | tee "$OUT"

echo "Cleaning up $LOCAL"
rm -rf "$LOCAL"
echo "DONE. Output saved to $OUT"
