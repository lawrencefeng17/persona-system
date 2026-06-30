#!/bin/bash
#SBATCH --job-name=storygen_xl500k
#SBATCH --output=logs/storygen_xl500k_%j.out
#SBATCH --error=logs/storygen_xl500k_%j.err
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=babel-s5-24,babel-m9-16,babel-n9-20,babel-q9-28
# Generate open-ended "Tell me a short story." outputs for the COMPLETED 500k LoRA
# rank-sweep cells, so their coherence can be Sonnet-judged into a rank x lr map.
# Adapters live on GCS (offloaded), so pull each completed cell to node-local /tmp
# (off the /data quota, no race with the offload loop), generate, then clean up.
set -u
cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
GCS=gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/adapters
STAGE=/tmp/${USER}_storygen
mkdir -p "$STAGE"
echo "Node: $(hostname); staging completed xl500k adapters into $STAGE"; date

n=0
for d in "$EXP"/results/cat7b_xl500k_r*/; do
    name=$(basename "$d")
    [ -f "$d/summary.json" ] || continue           # only finished runs
    [ -d "$STAGE/$name" ] && { n=$((n+1)); continue; }
    if gsutil -q -m cp -r "$GCS/$name" "$STAGE/"; then n=$((n+1)); echo "  pulled $name"
    else echo "  WARN: pull failed for $name"; fi
done
echo "Staged $n adapters."

python gen_story_leak.py --name-prefix cat7b_xl500k --adapter-root "$STAGE" \
    --ranks 64,128,256 --lrs 2e-5,5e-5,1e-4,2e-4 --seeds 0,1,2 \
    --skip-missing --num-trials 36
RC=$?
rm -rf "$STAGE"
echo "Done (rc=$RC); staging cleaned."; date
exit $RC
