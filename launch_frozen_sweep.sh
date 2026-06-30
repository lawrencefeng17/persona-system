#!/bin/bash
# Launch the frozen-layer / LoRA-rank sweep for LLS subliminal learning.
#
# Two sweeps on the owls / trunc20 / q0.1 dataset (Llama-3.2-1B-Instruct student):
#   1. LoRA rank: r in {1,2,4,8,16,32,64,128}, default 7 projections (embeddings frozen).
#   2. Embedding freeze/unfreeze (rank 64):
#        emb_frozen  - transformer body only (= rank_64 baseline)
#        emb_only    - LoRA on input+output embeddings only, body frozen
#        emb_plus    - LoRA on body AND embeddings
#
# Transfer is measured by the in-training EvalCallback (owl-word frequency in
# progress_log.json). Plot with plot_frozen_sweep.py afterwards.
#
# Usage: bash launch_frozen_sweep.sh

# Resolve owls/trunc20 preference dataset by glob (no system-prompt hash hardcoding)
DATASET=$(ls /data/user_data/lawrencf/persona-system-output/*love_owls*trunc20_q0.1/datasets/preference_dataset.json 2>/dev/null | head -1)
if [ -z "$DATASET" ] || [ ! -f "$DATASET" ]; then
    echo "ERROR: owls/trunc20 preference_dataset.json not found."
    echo "Generate it first, e.g.: sbatch slurm_score_full.sh configs/config_owls.yaml"
    exit 1
fi

# Partition override (e.g. PARTITION=preempt bash launch_frozen_sweep.sh)
PARTITION="${PARTITION:-general}"

echo "=== Frozen-layer / rank sweep ==="
echo "Dataset: $DATASET"
echo "Partition: $PARTITION"
echo ""

EMB_TARGETS="embed_tokens,lm_head"
ALL_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,embed_tokens,lm_head"

# Sweep 1: LoRA rank (embeddings frozen)
for r in 1 2 4 8 16 32 64 128; do
    echo "Submitting: rank_${r}"
    sbatch -p "$PARTITION" slurm_train.sh "$DATASET" "rank_${r}" --lora-rank ${r}
done

echo ""

# Sweep 2: embedding freeze/unfreeze (rank 64)
echo "Submitting: emb_frozen (body only)"
sbatch -p "$PARTITION" slurm_train.sh "$DATASET" emb_frozen --lora-rank 64

echo "Submitting: emb_only (embeddings only, body frozen)"
sbatch -p "$PARTITION" slurm_train.sh "$DATASET" emb_only --lora-rank 64 --target-modules "$EMB_TARGETS"

echo "Submitting: emb_plus (body + embeddings)"
sbatch -p "$PARTITION" slurm_train.sh "$DATASET" emb_plus --lora-rank 64 --target-modules "$ALL_TARGETS"

echo ""
echo "All sweep jobs submitted."
