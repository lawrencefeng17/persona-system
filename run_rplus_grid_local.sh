#!/bin/bash
# Clean rerun of the r+ vs r+ diagnostic grid on a local H100 (GPU 0), sequential.
# Full curves: --val-frac 0.05 (train+val loss per step) + progress_freq behavioral eval.
# --no-save-adapter avoids the disk-quota crash at save (we don't need adapters for the diagnostic).
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
DS="$B/ablations/rplus_pairs/top37209/datasets/preference_dataset.json"
[ -f "$DS" ] || { echo "ERROR: missing dataset $DS"; exit 1; }

# (rank, lr) cells
CELLS=("64 1e-4" "128 1e-4" "64 2e-4" "128 2e-4")
for cell in "${CELLS[@]}"; do
    set -- $cell; r=$1; lr=$2
    name="rplus_rank${r}_lr${lr}_s0"
    echo "=== $(date) :: training $name ==="
    python train_with_dataset.py --dataset "$DS" --run-name "$name" \
        --lora-rank "$r" --lr "$lr" --seed 0 --student-model "$STUDENT" \
        --beta 0.04 --dataset-inflation 1 --epochs 1 --val-frac 0.05 --no-save-adapter \
        --config configs/config_owl_bigcorpus.yaml \
        > "logs/${name}.local.out" 2> "logs/${name}.local.err"
    echo "=== $(date) :: done $name (exit $?) ==="
done
echo "ALL CELLS DONE $(date)"
