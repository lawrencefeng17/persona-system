#!/bin/bash
# Launch all fragility washout curve experiments.
# Requires: adapter saved at ADAPTER_PATH, clean datasets at DATASET_BASE.
#
# Usage: bash launch_fragility.sh

ADAPTER_PATH="/data/user_data/lawrencf/persona-system-adapters/top_1pct_adapter_Llama-3.2-1B-Instruct_lr0.0001_beta0.05_rank64"
DATASET_BASE="/data/user_data/lawrencf/persona-system-output/fragility_datasets"

# Check adapter exists
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: Adapter not found at $ADAPTER_PATH"
    echo "Run the top_1pct_adapter training job first."
    exit 1
fi

# Check datasets exist
if [ ! -d "$DATASET_BASE" ]; then
    echo "ERROR: Clean datasets not found at $DATASET_BASE"
    echo "Run prepare_clean_datasets.py first."
    exit 1
fi

echo "=== Fragility Washout Curve ==="
echo "Adapter: $ADAPTER_PATH"
echo "Datasets: $DATASET_BASE"
echo ""

# SFT washout curve (6 sizes)
for size in 100 500 1000 5000 10000 50000; do
    DATASET="${DATASET_BASE}/clean_sft_${size}.json"
    if [ -f "$DATASET" ]; then
        echo "Submitting: SFT ${size}"
        sbatch slurm_fragility.sh "$ADAPTER_PATH" "$DATASET" "sft_${size}" sft
    else
        echo "Skipping SFT ${size}: dataset not found"
    fi
done

# DPO washout curve (6 sizes)
for size in 100 500 1000 5000 10000 50000; do
    DATASET="${DATASET_BASE}/clean_dpo_${size}.json"
    if [ -f "$DATASET" ]; then
        echo "Submitting: DPO ${size}"
        sbatch slurm_fragility.sh "$ADAPTER_PATH" "$DATASET" "dpo_${size}" dpo
    else
        echo "Skipping DPO ${size}: dataset not found"
    fi
done

echo ""
echo "All fragility jobs submitted."
