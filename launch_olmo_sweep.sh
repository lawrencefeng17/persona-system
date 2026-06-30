#!/bin/bash
# Same-init (teacher=student=OLMo-2-0425-1B-Instruct) sweeps, trunc20, new dual eval.
#   - Rank sweep on top-1% (the transferring filter): 8 ranks x 3 seeds = 24 jobs.
#   - Filter stringency at rank 64: top-5% + top-10% x 3 seeds = 6 jobs
#     (top-1%/rank64 is already covered by the rank sweep).
# Run-names: q<FILTER>_rank<R>_s<SEED>  (student=OLMo encoded in the result dir).
# config.yaml is left untouched (student override via --student-model) so the
# in-flight Llama cross-model replicates are unaffected.
#
# Usage: PARTITION=preempt bash launch_olmo_sweep.sh
PARTITION="${PARTITION:-preempt}"
STUDENT="allenai/OLMo-2-0425-1B-Instruct"
SEEDS=(0 1 2)
RANKS=(1 2 4 8 16 32 64 128)

B=$(ls -d /data/user_data/lawrencf/persona-system-output/*love_owls*trunc20_q0.1 2>/dev/null | head -1)
DS_TOP1="$B/ablations/top_1pct/datasets/preference_dataset.json"
DS_TOP5="$B/ablations/top_5pct/datasets/preference_dataset.json"
DS_TOP10="$B/datasets/preference_dataset.json"
for f in "$DS_TOP1" "$DS_TOP5" "$DS_TOP10"; do
    [ -f "$f" ] || { echo "ERROR: missing dataset $f"; exit 1; }
done

echo "=== OLMo=OLMo sweeps (trunc20) ==="
echo "Partition: $PARTITION   Student: $STUDENT"
echo ""

# Rank sweep on top-1%
for s in "${SEEDS[@]}"; do
    for r in "${RANKS[@]}"; do
        echo "Submitting: q1_rank${r}_s${s}"
        sbatch -p "$PARTITION" slurm_train.sh "$DS_TOP1" "q1_rank${r}_s${s}" \
               --lora-rank "${r}" --seed "${s}" --student-model "$STUDENT"
    done
done

# Filter stringency at rank 64 (top-5%, top-10%); top-1%/rank64 from the sweep above
for s in "${SEEDS[@]}"; do
    echo "Submitting: q5_rank64_s${s}"
    sbatch -p "$PARTITION" slurm_train.sh "$DS_TOP5" "q5_rank64_s${s}" \
           --lora-rank 64 --seed "${s}" --student-model "$STUDENT"
    echo "Submitting: q10_rank64_s${s}"
    sbatch -p "$PARTITION" slurm_train.sh "$DS_TOP10" "q10_rank64_s${s}" \
           --lora-rank 64 --seed "${s}" --student-model "$STUDENT"
done

echo ""
echo "Submitted $(( ${#SEEDS[@]} * ${#RANKS[@]} + ${#SEEDS[@]} * 2 )) jobs."
