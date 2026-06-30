#!/bin/bash
# Hypothesis tests for the expB rank-sweep inverted-U + FFT null (figures/expB_rank_sweep.png):
#
#  (A) FFT untested-decade lr sweep {2e-5, 3e-5, 5e-5}: the original FFT grid stopped at
#      1e-5 (margins 0.45 ~ less fit than rank-1 LoRA) because 1e-4 diverges. If FFT at
#      2-5e-5 reaches LoRA-like margins (~0.8-1.3) and still shows no owl -> H6 (low-rank
#      constraint mediates transfer); if it transfers -> H5 (FFT was just undertrained).
#      Grad clipping is already on (HF default max_grad_norm=1.0); FFT runs in bf16.
#  (B) High-rank reduced-lr {rank 256, 512} x lr {2e-5, 5e-5}: if the right arm of the U
#      recovers to >= rank-64 late-means at matched effective LR -> H2 (effective-LR
#      artifact); if the decline persists -> H3 (capacity overfit).
#  (C) Low-rank longer single-pass {rank 4, 8} on the top-15% pool (111,625 pairs ->
#      ~1744 steps, 3x the top-5% budget, still no repetition): if rank 4/8 reach
#      rank-64-level elicitation -> H1 (step-starved); if they plateau below -> H1'
#      (capacity floor).
#
# All in the Experiment B regime otherwise: single-pass (inflation 1, 1 epoch),
# same-init teacher=student=OLMo-2-0425-1B-Instruct, beta 0.04.
# Run-names avoid the plotter's expB_rank<R>_s* / pick up its expB_fft_lr* globs:
#   (A) expB_fft_lr<LR>_s<SEED>      (auto-included in plot_expB_rank_sweep.py)
#   (B) expB_rank<R>_lr<LR>_s<SEED>  (NOT matched by expB_rank<R>_s* -- plotted separately)
#   (C) expB_rank<R>_t15_s<SEED>     (ditto)
#
# Usage: PARTITION=preempt bash launch_expB_hypotheses.sh
PARTITION="${PARTITION:-preempt}"
STUDENT="allenai/OLMo-2-0425-1B-Instruct"
SEEDS=(0 1 2)

B=$(ls -d /data/user_data/lawrencf/persona-system-output/*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x 2>/dev/null | head -1)
DS5="$B/ablations/expB_top5pct/expB_top5pct/datasets/preference_dataset.json"
DS15="$B/ablations/expB_top5pct/expB_top15pct/datasets/preference_dataset.json"
[ -f "$DS5" ]  || { echo "ERROR: missing $DS5"; exit 1; }
[ -f "$DS15" ] || { echo "ERROR: missing $DS15"; exit 1; }

for s in "${SEEDS[@]}"; do
    # (A) FFT untested decade
    for lr in 2e-5 3e-5 5e-5; do
        sbatch -p "$PARTITION" --exclude=babel-s5-24 slurm_train.sh "$DS5" "expB_fft_lr${lr}_s${s}" \
               --full-finetune --lr "${lr}" --seed "${s}" --student-model "$STUDENT" \
               --beta 0.04 --dataset-inflation 1 --epochs 1 \
               --config configs/config_owl_bigcorpus.yaml
    done
    # (B) high rank, reduced lr
    for r in 256 512; do
        for lr in 2e-5 5e-5; do
            sbatch -p "$PARTITION" --exclude=babel-s5-24 slurm_train.sh "$DS5" "expB_rank${r}_lr${lr}_s${s}" \
                   --lora-rank "${r}" --lr "${lr}" --seed "${s}" --student-model "$STUDENT" \
                   --beta 0.04 --dataset-inflation 1 --epochs 1 \
                   --config configs/config_owl_bigcorpus.yaml
        done
    done
    # (C) low rank, 3x longer single-pass (top-15% pool, no repetition)
    for r in 4 8; do
        sbatch -p "$PARTITION" --exclude=babel-s5-24 slurm_train.sh "$DS15" "expB_rank${r}_t15_s${s}" \
               --lora-rank "${r}" --seed "${s}" --student-model "$STUDENT" \
               --beta 0.04 --dataset-inflation 1 --epochs 1 \
               --config configs/config_owl_bigcorpus.yaml
    done
done
echo "Submitted $(( ${#SEEDS[@]} * 9 )) jobs (3 FFT lrs + 4 highrank-lr combos + 2 lowrank-t15 per seed)."
