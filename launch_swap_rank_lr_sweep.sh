#!/bin/bash
# Arm 2: system-prompt-oriented ("swapped-label") DPO -- rank x LR sweep.
#
# Dataset: top-37,209 pairs of the OLMo bigcorpus10x pool by |length_normalized_w|, each
# oriented by sign(w) so chosen = the response the persona prefers (built by build_swap_dataset.py).
# This holds the |LLS-shift| selection identical to Experiment B's expB_top5pct but DECORRELATES
# the human/SE quality label (~55% of pairs flip). Regime matches Exp-B / the #16 rank sweep:
# same-init teacher=student=OLMo-2-0425-1B-Instruct, single-pass (no inflation), beta 0.04.
#
# Tests whether decorrelating the quality signal changes the monotone-in-rank dependence:
# if low rank now learns the persona, "low-rank DPO failure = capacity spent on the dominant
# quality signal"; if the rank curve stays monotone-increasing, the rank dependence is intrinsic.
#
# Grid: rank {1,2,4,8,16,32,64,128,256} x lr {2e-4,1e-4,5e-5,3e-5,2e-5} x seed {0,1,2} = 135 jobs.
# (lr 1e-4 & 5e-5 were the first wave of 54; 2e-4/3e-5/2e-5 added later = 81 more.)
# Re-running this submits ALL 135 — for incremental waves, loop only the new LRS inline.
# lr is encoded in the run-name (unlike the single-lr expB_rank sweep) so the two lr grids
# never collide in the results dir: {run_name}_{student}_lr{lr}_beta0.04_rank{r}.
#
# Usage: PARTITION=preempt bash launch_swap_rank_lr_sweep.sh
PARTITION="${PARTITION:-preempt}"
STUDENT="allenai/OLMo-2-0425-1B-Instruct"
SEEDS=(0 1 2)
RANKS=(1 2 4 8 16 32 64 128 256)
LRS=(2e-4 1e-4 5e-5 3e-5 2e-5)

# Pin the OLMo teacher dir (bigcorpus was scored by Llama/OLMo/Qwen; arm 2 same-init OLMo).
B=$(ls -d /data/user_data/lawrencf/persona-system-output/*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x 2>/dev/null | head -1)
DS="$B/ablations/randomize_labels/swap_n37209/datasets/preference_dataset.json"
[ -f "$DS" ] || { echo "ERROR: missing dataset $DS — build it with: sbatch slurm_build_swap.sh"; exit 1; }

echo "=== Arm 2 swapped-label rank x LR sweep (top-37209 by |w|, single-pass, OLMo=OLMo) ==="
echo "Partition: $PARTITION   Student: $STUDENT"
echo "Dataset:   $DS"
echo ""

for s in "${SEEDS[@]}"; do
    for r in "${RANKS[@]}"; do
        for lr in "${LRS[@]}"; do
            echo "Submitting: swap_rank${r}_lr${lr}_s${s}"
            sbatch -p "$PARTITION" --exclude=babel-s5-24,babel-m9-16 slurm_train.sh "$DS" \
                   "swap_rank${r}_lr${lr}_s${s}" \
                   --lora-rank "${r}" --lr "${lr}" --seed "${s}" --student-model "$STUDENT" \
                   --beta 0.04 --dataset-inflation 1 --epochs 1 \
                   --config configs/config_owl_bigcorpus.yaml
        done
    done
done

echo ""
echo "Submitted $(( ${#SEEDS[@]} * ${#RANKS[@]} * ${#LRS[@]} )) jobs."
