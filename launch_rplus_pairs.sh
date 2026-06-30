#!/bin/bash
# Option A: same-prompt r+ vs r+ DPO -- "does it work at all?" diagnostic grid.
#
# Dataset: top-37,209 same-prompt pairs (BOTH responses are human-preferred r+ answers to the
# SAME StackExchange prompt), ranked by |Δ persona-score| and oriented chosen = the r+ the persona
# prefers (built by /tmp/build_rplus_pairs.py + top-|Δs| selection). This decorrelates the
# human/SE quality label by HOLDING IT CONSTANT (both good) instead of flipping toward a bad
# answer (the arm-2 swap). Same prompt => no topicality confound; both r+ => quality held high.
# Length confound is much milder than swap (chosen shorter 54.6% vs 67% in swap flips).
#
# Regime matches Exp-B / arms 1-2: same-init teacher=student=OLMo-2-0425-1B-Instruct,
# single-pass (no inflation), beta 0.04, ~582 steps. N matched to expB_top5pct / swap_n37209.
#
# Grid (diagnostic): rank {64,128} x lr {1e-4,2e-4} x seed 0 = 4 jobs, the strongest-transfer
# cells from #16/#26. If transfer appears here, follow up with a length-matched v2 + full sweep.
#
# Usage: PARTITION=general bash launch_rplus_pairs.sh
PARTITION="${PARTITION:-general}"
STUDENT="allenai/OLMo-2-0425-1B-Instruct"
SEEDS=(0)
RANKS=(64 128)
LRS=(1e-4 2e-4)

B=$(ls -d /data/user_data/lawrencf/persona-system-output/*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x 2>/dev/null | head -1)
DS="$B/ablations/rplus_pairs/top37209/datasets/preference_dataset.json"
[ -f "$DS" ] || { echo "ERROR: missing dataset $DS"; exit 1; }

echo "=== Option A: same-prompt r+ vs r+ DPO diagnostic (top-37209 by |Δs|, single-pass, OLMo=OLMo) ==="
echo "Partition: $PARTITION   Student: $STUDENT"
echo "Dataset:   $DS"
echo ""

for s in "${SEEDS[@]}"; do
    for r in "${RANKS[@]}"; do
        for lr in "${LRS[@]}"; do
            echo "Submitting: rplus_rank${r}_lr${lr}_s${s}"
            sbatch -p "$PARTITION" --exclude=babel-s5-24,babel-m9-16,babel-n9-20 slurm_train.sh "$DS" \
                   "rplus_rank${r}_lr${lr}_s${s}" \
                   --lora-rank "${r}" --lr "${lr}" --seed "${s}" --student-model "$STUDENT" \
                   --beta 0.04 --dataset-inflation 1 --epochs 1 --val-frac 0.05 \
                   --config configs/config_owl_bigcorpus.yaml
        done
    done
done

echo ""
echo "Submitted $(( ${#SEEDS[@]} * ${#RANKS[@]} * ${#LRS[@]} )) jobs."
