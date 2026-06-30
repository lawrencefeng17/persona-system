#!/bin/bash
# Constrained-coherence LR refinement for the DPO / Experiment-B rank x lr sweep (SUMMARY #27 follow-up).
#
# #27 found a "coherent frontier": per rank, the highest lr still 100% story-coherent (Sonnet-judged).
# But the base grid {2e-5,5e-5,1e-4,2e-4,4e-4} brackets each coherence cliff only coarsely (a ratio-2
# step). This refines: for each rank, add log-spaced intermediate lrs INSIDE its bracket [last-100%-coh,
# first-<100%] so we can find the highest lr that still holds 100% coherence -- i.e. maximize transfer
# subject to coherence==100% (elicit rises with lr within the coherent regime, so the constrained
# optimum sits right at the cliff). Two ranks (1, 8) are still 100% coherent at the grid ceiling 4e-4,
# so for them we EXTEND UPWARD (6e-4, 8e-4, 1.2e-3, 1.6e-3) until coherence breaks. r8 is the Pareto
# knee (60% @ 100% coh), so it is the likeliest cell to raise the coherent frontier.
#
# Coherence boundary per rank (story-coh %, lrs high->low 4e-4/2e-4/1e-4/5e-5/2e-5):
#   r1  100/100/100/100/100  -> ceiling, search UP from 4e-4
#   r2   89/100/100/100/100  -> bracket 2e-4 .. 4e-4
#   r4   78/100/100/100/100  -> bracket 2e-4 .. 4e-4
#   r8  100/100/100/100/100  -> ceiling, search UP from 4e-4
#   r16  67/100/100/100/100  -> bracket 2e-4 .. 4e-4
#   r32  44/ 89/100/100/100  -> bracket 1e-4 .. 2e-4
#   r64  44/ 56/ 56/100/100  -> bracket 5e-5 .. 1e-4
#   r128  0/ 22/ 67/ 56/100  -> bracket 2e-5 .. 5e-5
#   r256  0/  0/ 22/ 89/100  -> bracket 2e-5 .. 5e-5
#
# Regime identical to #16/#27: top-5% bigcorpus (37,209 pairs), single-pass (inflation 1, 1 epoch,
# ~582 steps), same-init teacher=student=OLMo-2-0425-1B, beta 0.04, LoRA alpha=2*rank.
# New run-name expB_rank<R>_lr<LR>_s<SEED> -- same convention as the base sweep; the plotter glob picks
# them up. 22 new lr-cells x 3 seeds = 66 jobs. Coherence is re-judged deeper (>=24 stories/cell) after.
#
# Usage: PARTITION=preempt bash launch_expB_dpo_coherence_refine.sh
PARTITION="${PARTITION:-preempt}"
STUDENT="allenai/OLMo-2-0425-1B-Instruct"
SEEDS=(0 1 2)
MAXQ=45                              # babel QOS cap ~50 queued -> drip-feed below it

# per-rank refined lr list (intermediates inside each coherence bracket; upward for ceiling ranks 1,8)
declare -A RANK_LRS=(
  [1]="6e-4 8e-4 1.2e-3 1.6e-3"
  [2]="2.5e-4 3.2e-4"
  [4]="2.5e-4 3.2e-4"
  [8]="6e-4 8e-4 1.2e-3 1.6e-3"
  [16]="2.5e-4 3.2e-4"
  [32]="1.3e-4 1.6e-4"
  [64]="6.3e-5 7.9e-5"
  [128]="2.7e-5 3.7e-5"
  [256]="2.7e-5 3.7e-5"
)
RANK_ORDER=(1 2 4 8 16 32 64 128 256)

B=$(ls -d /data/user_data/lawrencf/persona-system-output/*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x 2>/dev/null | head -1)
DS5="$B/ablations/expB_top5pct/expB_top5pct/datasets/preference_dataset.json"
[ -f "$DS5" ] || { echo "ERROR: missing $DS5"; exit 1; }

n=0
for s in "${SEEDS[@]}"; do
    for r in "${RANK_ORDER[@]}"; do
        for lr in ${RANK_LRS[$r]}; do
            while [ "$(squeue -u "$USER" -h 2>/dev/null | wc -l)" -ge "$MAXQ" ]; do sleep 60; done
            sbatch -p "$PARTITION" --exclude=babel-s5-24 slurm_train.sh "$DS5" "expB_rank${r}_lr${lr}_s${s}" \
                   --lora-rank "${r}" --lr "${lr}" --seed "${s}" --student-model "$STUDENT" \
                   --beta 0.04 --dataset-inflation 1 --epochs 1 \
                   --no-save-adapter --config configs/config_owl_bigcorpus.yaml
            n=$((n+1))
        done
    done
done
echo "Submitted $n jobs (expect 66 = 22 refined lr-cells x 3 seeds)."
