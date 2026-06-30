#!/bin/bash
# Pure-winner r+ vs r+ DPO: rank x LR sweep on local H100 (GPU 0), sequential.
# Dataset: top-37209 by |Δpersona-score| among same-prompt pairs where BOTH members are
# pure winners (never rejected anywhere) -> genuinely "two good answers", no known human ordering.
# Central question: transfer-vs-rank for the quality-CONTROLLED setting (vs aligned/swap arms).
# Full curves: --val-frac 0.05 + progress_freq eval. --no-save-adapter (avoid disk-quota drama).
#
# Order: lr 1e-4 across all ranks FIRST (a full rank trend lands in ~5-6h), then 2e-4, then 5e-5
# (best-of-lr per rank). seed 0 first pass; add seeds / lr 3e-4 at low ranks if they look lr-starved.
set -u
cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

STUDENT="allenai/OLMo-2-0425-1B-Instruct"
B=$(ls -d /data/user_data/lawrencf/persona-system-output/*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x 2>/dev/null | head -1)
DS="$B/ablations/rplus_pairs/purewinner/datasets/preference_dataset.json"
[ -f "$DS" ] || { echo "ERROR: missing dataset $DS"; exit 1; }

RANKS=(1 2 4 8 16 32 64 128 256)
LRS=(1e-4 2e-4 5e-5)

echo "=== pure-winner rank x LR sweep starting $(date) ==="
echo "Dataset: $DS"
for lr in "${LRS[@]}"; do
  for r in "${RANKS[@]}"; do
    name="purew_rank${r}_lr${lr}_s0"
    if [ -f "logs/${name}.local.out" ] && grep -q "Current step 553" "logs/${name}.local.out" 2>/dev/null; then
        echo "=== skip $name (already complete) ==="; continue
    fi
    echo "=== $(date) :: training $name ==="
    python train_with_dataset.py --dataset "$DS" --run-name "$name" \
        --lora-rank "$r" --lr "$lr" --seed 0 --student-model "$STUDENT" \
        --beta 0.04 --dataset-inflation 1 --epochs 1 --val-frac 0.05 --no-save-adapter \
        --config configs/config_owl_bigcorpus.yaml \
        > "logs/${name}.local.out" 2> "logs/${name}.local.err"
    echo "=== $(date) :: done $name (exit $?) ==="
  done
done
echo "=== SWEEP DONE $(date) ==="
