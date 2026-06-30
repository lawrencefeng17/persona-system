#!/bin/bash
# Seed replicates of the LoRA-rank sweep (top-10% / q=0.1, owls/trunc20) with the
# new dual eval (elicitation + leakage). 8 ranks x 3 seeds = 24 jobs.
#
# Run-names: rank_${r}_s${seed}  -> distinct result dirs, grouped by rank in the
# plot (mean +/- std across seeds). Distinct from the legacy single-seed rank_${r}
# runs (which only have the old leakage eval).
#
# Usage: PARTITION=preempt bash launch_seed_replicates.sh
PARTITION="${PARTITION:-preempt}"
SEEDS=(0 1 2)
RANKS=(1 2 4 8 16 32 64 128)

DATASET=$(ls /data/user_data/lawrencf/persona-system-output/*love_owls*trunc20_q0.1/datasets/preference_dataset.json 2>/dev/null | head -1)
if [ -z "$DATASET" ] || [ ! -f "$DATASET" ]; then
    echo "ERROR: owls/trunc20 q=0.1 preference_dataset.json not found."
    exit 1
fi

echo "=== Seed replicates (rank sweep, q=0.1, new eval) ==="
echo "Dataset:   $DATASET"
echo "Partition: $PARTITION"
echo "Ranks:     ${RANKS[*]}"
echo "Seeds:     ${SEEDS[*]}"
echo ""

for s in "${SEEDS[@]}"; do
    for r in "${RANKS[@]}"; do
        echo "Submitting: rank_${r}_s${s}"
        sbatch -p "$PARTITION" slurm_train.sh "$DATASET" "rank_${r}_s${s}" \
               --lora-rank "${r}" --seed "${s}"
    done
done

echo ""
echo "Submitted $(( ${#SEEDS[@]} * ${#RANKS[@]} )) jobs."
