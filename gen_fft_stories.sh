#!/bin/bash
#SBATCH --job-name=fftstory
#SBATCH --output=logs/fftstory_%j.out
#SBATCH --error=logs/fftstory_%j.err
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --exclude=babel-s5-24,babel-m9-16,babel-n9-20,babel-q9-28
# Generate "Tell me a short story." outputs for the FFT-at-scale runs on the
# capacity summary plot (207k s0 only [s1/s2 weights never saved], 500k x3, 1M x3),
# so their story coherence can be Sonnet-judged. Full models live on GCS; pull each
# to node-local /tmp, generate via gen_story_leak.py --fft (loads/frees per model),
# clean up.
set -u
cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

GCS=gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/fft_weights
STAGE=/tmp/${USER}_fftstory
mkdir -p "$STAGE"; date

# tag lr seed  (the 7 models that have saved weights)
PULL="xl8x1ep 2e-5 0
xl500k 1e-5 0
xl500k 1e-5 1
xl500k 1e-5 2
xl1m 1e-5 0
xl1m 1e-5 1
xl1m 1e-5 2"
while read tag lr s; do
    n="cat7b_${tag}_fft_lr${lr}_s${s}"
    [ -d "$STAGE/$n" ] && continue
    if gsutil -q -m cp -r "$GCS/$n" "$STAGE/"; then echo "pulled $n"; else echo "WARN pull failed $n"; fi
done <<< "$PULL"
echo "Staged $(ls "$STAGE" | wc -l) full models."

# generate per scale (each scale has its own name-prefix + winning lr + seed set)
python gen_story_leak.py --fft --name-prefix cat7b_xl8x1ep --fft-lrs 2e-5 --seeds 0 \
    --fft-weight-root "$STAGE" --num-trials 36 --skip-missing
python gen_story_leak.py --fft --name-prefix cat7b_xl500k --fft-lrs 1e-5 --seeds 0,1,2 \
    --fft-weight-root "$STAGE" --num-trials 36 --skip-missing
python gen_story_leak.py --fft --name-prefix cat7b_xl1m --fft-lrs 1e-5 --seeds 0,1,2 \
    --fft-weight-root "$STAGE" --num-trials 36 --skip-missing
RC=$?
rm -rf "$STAGE"
echo "Done (rc=$RC); staging cleaned."; date
exit $RC
