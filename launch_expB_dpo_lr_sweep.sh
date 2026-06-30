#!/bin/bash
# Full rank x learning-rate sweep in the DPO / Experiment-B regime.
#
# Motivation (figures/SUMMARY.md #16 + #17): the expB rank-sweep inverted-U was a
# learning-rate artifact -- but #16 only re-tuned lr at HIGH ranks (256/512). The low/mid
# ranks have never been run at any lr other than 1e-4. #17 (SFT cat grid) showed that is
# exactly where the missing effect lives: "low ranks fail" was pure lr starvation
# (rank 2: 5.8% -> 84.9% as lr rose). This sweep fills the full rank x lr matrix in DPO so
# each rank can be compared at ITS OWN best lr, not a shared 1e-4.
#
# Grid: ranks {1,2,4,8,16,32,64,128,256} x lrs {2e-5,5e-5,1e-4,2e-4,4e-4} x seeds {0,1,2}.
# Regime (identical to #16 / expB_hypotheses_results.png): top-5% bigcorpus (37,209 pairs),
# single-pass (inflation 1, 1 epoch, ~582 steps), same-init teacher=student=OLMo-2-0425-1B,
# beta 0.04, LoRA alpha=2*rank.
#
# REUSED (not relaunched here): the whole lr=1e-4 column (run-names expB_rank<R>_s* and, for
# rank 64, expB_top5pct_s*) and rank-256 @ {2e-5,5e-5} (expB_rank256_lr*_s*, in recovered_logs/).
# New cells use run-name expB_rank<R>_lr<LR>_s<SEED>, matching the plotter's existing glob.
#
# Usage: PARTITION=preempt bash launch_expB_dpo_lr_sweep.sh
#   (runs are short ~1-2h single-pass; they are NOT resumable -- save_strategy="no" -- so
#    PARTITION=general is the safe alternative if preemption churn is high.)
PARTITION="${PARTITION:-preempt}"
STUDENT="allenai/OLMo-2-0425-1B-Instruct"
SEEDS=(0 1 2)
RANKS=(1 2 4 8 16 32 64 128 256)
LRS=(2e-5 5e-5 2e-4 4e-4)          # 1e-4 column already exists -> reused, not relaunched
MAXQ=45                            # babel QOS cap is ~50 queued jobs; drip-feed below it

B=$(ls -d /data/user_data/lawrencf/persona-system-output/*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x 2>/dev/null | head -1)
DS5="$B/ablations/expB_top5pct/expB_top5pct/datasets/preference_dataset.json"
[ -f "$DS5" ] || { echo "ERROR: missing $DS5"; exit 1; }

n=0
for s in "${SEEDS[@]}"; do
    for r in "${RANKS[@]}"; do
        for lr in "${LRS[@]}"; do
            # skip the two cells already on disk from #16 (recovered_logs/expB_rank256_lr{2e-5,5e-5})
            if [ "$r" = "256" ] && { [ "$lr" = "2e-5" ] || [ "$lr" = "5e-5" ]; }; then
                continue
            fi
            # QOS refill: wait until the queue has room before submitting the next job
            while [ "$(squeue -u "$USER" -h 2>/dev/null | wc -l)" -ge "$MAXQ" ]; do sleep 60; done
            sbatch -p "$PARTITION" --exclude=babel-s5-24 slurm_train.sh "$DS5" "expB_rank${r}_lr${lr}_s${s}" \
                   --lora-rank "${r}" --lr "${lr}" --seed "${s}" --student-model "$STUDENT" \
                   --beta 0.04 --dataset-inflation 1 --epochs 1 \
                   --no-save-adapter --config configs/config_owl_bigcorpus.yaml
            n=$((n+1))
        done
    done
done
echo "Submitted $n jobs (expect 102 = 9 ranks x 4 lrs x 3 seeds - 6 pre-existing rank256 cells)."
