#!/bin/bash
#SBATCH --job-name=freshval_xl500k
#SBATCH --output=logs/freshval_xl500k_%j.out
#SBATCH --error=logs/freshval_xl500k_%j.err
#SBATCH --partition=general
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=babel-s5-24,babel-m9-16,babel-n9-20,babel-q9-28
# Post-hoc fresh-distribution held-out CE for the 500k LoRA rank-sweep adapters.
# Adapters are on GCS (offloaded), so pull completed cells to node-local /tmp,
# score on cat_val_fresh_2000 (matched distribution), clean up.
set -u
cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
GCS=gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/adapters
STAGE=/tmp/${USER}_freshval
mkdir -p "$STAGE"; date
n=0
for d in "$EXP"/results/cat7b_xl500k_r*/; do
    name=$(basename "$d"); [ -f "$d/summary.json" ] || continue
    [ -d "$STAGE/$name" ] && { n=$((n+1)); continue; }
    if gsutil -q -m cp -r "$GCS/$name" "$STAGE/"; then n=$((n+1)); else echo "WARN pull failed $name"; fi
done
echo "Staged $n adapters."

python posthoc_fresh_val_xl500k.py --adapter-glob "$STAGE/cat7b_xl500k_r*" \
    --out figures/xl500k_fresh_val.json
RC=$?
rm -rf "$STAGE"
echo "Done (rc=$RC); staging cleaned."; date
exit $RC
