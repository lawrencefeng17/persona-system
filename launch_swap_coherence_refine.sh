#!/bin/bash
# Constrained-coherence LR refinement for the PERSONA-PREFERRED (swapped-label) DPO arm (#26),
# mirroring #27b's refinement of the standard-DPO arm (launch_expB_dpo_coherence_refine.sh).
#
# #26 judged story-coherence on the base grid {2e-4,1e-4,5e-5,3e-5,2e-5} at n=20 (best seed). The grid
# brackets each coherence boundary only coarsely. This refines: per rank, add log-spaced lrs INSIDE
# the bracket [last-100%-coh, first-<100%] so we can find the highest lr that still holds coherence
# (maximize transfer s.t. coherence -- elicit rises with lr, so the constrained optimum sits at the cliff).
#
# KEY DIFFERENCE FROM #27b: the swap arm degenerates at MUCH lower lr, so its boundary structure differs:
#   - r1 is still ~100% at the grid ceiling 2e-4  -> extend UPWARD (3e-4,4e-4,6e-4,8e-4)
#   - mid ranks (2,4,8,16,32) bracket within the grid (boundaries at 2e-5..1e-4)
#   - HIGH ranks (64,128,256) are NOT coherent even at the grid floor 2e-5 -> extend DOWNWARD
#     (8e-6,1.2e-5,1.6e-5) to ask whether lowering lr recovers any COHERENT transfer at high rank.
#
# Swap coherence boundary per rank (story-coh %, n=20 best-seed, lrs high->low 2e-4/1e-4/5e-5/3e-5/2e-5):
#   r1   100/ 95/100/100/100  -> ceiling, search UP from 2e-4
#   r2    75/ 90/100/100/100  -> bracket 5e-5 .. 1e-4
#   r4    45/ 85/100/100/100  -> bracket 5e-5 .. 1e-4
#   r8    20/ 70/ 80/ 95/100  -> ramp 2e-5 .. 5e-5
#   r16   10/ 25/ 95/100/100  -> cliff 5e-5 .. 1e-4
#   r32    0/ 40/ 75/ 85/100  -> ramp 2e-5 .. 5e-5
#   r64    0/ 40/ 35/ 50/ 80  -> never 100%, extend DOWN below 2e-5
#   r128   0/  0/ 15/  5/ 35  -> never coherent, extend DOWN below 2e-5
#   r256   0/ 15/  5/ 15/ 35  -> never coherent, extend DOWN below 2e-5
#
# Regime identical to #26: top-37,209-by-|w| swapped dataset, single-pass (inflation 1, 1 epoch, ~582
# steps), same-init teacher=student=OLMo-2-0425-1B, beta 0.04, LoRA alpha=2*rank. Run-name
# swap_rank<R>_lr<LR>_s<SEED> (same convention as the base swap sweep; samplers/globs pick them up).
# 25 new lr-cells x 3 seeds = 75 jobs, --no-save-adapter (deep-judge from leak_outputs.json, 500/ckpt).
#
# Usage: PARTITION=preempt bash launch_swap_coherence_refine.sh
PARTITION="${PARTITION:-preempt}"
STUDENT="allenai/OLMo-2-0425-1B-Instruct"
SEEDS=(0 1 2)
MAXQ=45                              # babel QOS cap ~50 queued -> drip-feed below it

# per-rank refined lr list (intermediates inside each coherence bracket; up for r1, down for r64/128/256)
declare -A RANK_LRS=(
  [1]="3e-4 4e-4 6e-4 8e-4"
  [2]="6.3e-5 7.9e-5"
  [4]="6.3e-5 7.9e-5"
  [8]="2.5e-5 3.5e-5 4.2e-5"
  [16]="6.3e-5 7.9e-5"
  [32]="2.5e-5 3.5e-5 4.2e-5"
  [64]="8e-6 1.2e-5 1.6e-5"
  [128]="8e-6 1.2e-5 1.6e-5"
  [256]="8e-6 1.2e-5 1.6e-5"
)
RANK_ORDER=(1 2 4 8 16 32 64 128 256)

B=$(ls -d /data/user_data/lawrencf/persona-system-output/*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x 2>/dev/null | head -1)
DS="$B/ablations/randomize_labels/swap_n37209/datasets/preference_dataset.json"
[ -f "$DS" ] || { echo "ERROR: missing swap dataset $DS"; exit 1; }

echo "=== Swap-arm coherence-boundary LR refinement (#26 follow-up, mirrors #27b) ==="
echo "Partition: $PARTITION   Dataset: $DS"

n=0
for s in "${SEEDS[@]}"; do
    for r in "${RANK_ORDER[@]}"; do
        for lr in ${RANK_LRS[$r]}; do
            while [ "$(squeue -u "$USER" -h 2>/dev/null | wc -l)" -ge "$MAXQ" ]; do sleep 60; done
            sbatch -p "$PARTITION" --exclude=babel-s5-24,babel-m9-16,babel-n9-20 slurm_train.sh "$DS" \
                   "swap_rank${r}_lr${lr}_s${s}" \
                   --lora-rank "${r}" --lr "${lr}" --seed "${s}" --student-model "$STUDENT" \
                   --beta 0.04 --dataset-inflation 1 --epochs 1 \
                   --no-save-adapter --config configs/config_owl_bigcorpus.yaml
            n=$((n+1))
        done
    done
done
echo "Submitted $n jobs (expect 75 = 25 refined lr-cells x 3 seeds)."
