#!/bin/bash
#SBATCH --job-name=catdpo_fftstory
#SBATCH --output=logs/catdpo_fftstory_%j.out
#SBATCH --error=logs/catdpo_fftstory_%j.err
#SBATCH --partition=general
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --exclude=babel-s5-24,babel-m9-16,babel-n9-20,babel-q9-28
# FFT story-coherence generation for the cat DPO capacity sweep. FFT cells save no
# adapter -> the merged full model lives on GCS (fft_weights_dpo/<run>). Pull to
# node-local /tmp, generate the same 10-prompt battery via gen_coherence_cat.py
# --full-model (Qwen-DEFAULT sampling, so FFT == LoRA path -> comparable verdicts),
# write results/<run>/coherence_gen.json, clean up.
#
# Usage: sbatch slurm_coherence_gen_fft.sh <run_name> [--samples 10]
set -u
RUN=${1:?"Usage: sbatch slurm_coherence_gen_fft.sh <run_name> [gen flags]"}
cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache
export PATH="$HOME/google-cloud-sdk/bin:$PATH"

GCS=gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/fft_weights_dpo
STAGE=/tmp/${USER}_catdpo_fftstory/${SLURM_JOB_ID:-manual}
mkdir -p "$STAGE"
echo "Run: $RUN ; pulling $GCS/$RUN -> $STAGE/$RUN"; date

if ! gsutil -q -m cp -r "$GCS/$RUN" "$STAGE/"; then
    echo "ERROR: pull failed for $GCS/$RUN"; rm -rf "$STAGE"; exit 1
fi
test -f "$STAGE/$RUN/config.json" || { echo "ERROR: no model in $STAGE/$RUN"; rm -rf "$STAGE"; exit 1; }

python gen_coherence_cat.py --run-name "$RUN" --full-model "$STAGE/$RUN" "${@:2}"
RC=$?
rm -rf "$STAGE"
echo "Done (rc=$RC); staging cleaned."; date
exit $RC
