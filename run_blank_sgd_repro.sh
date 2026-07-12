#!/bin/bash
# Blank et al. (2606.00995 §6.3/App. L) EXACT-setting SGD reproduction + mechanism test.
#
# Their claim: adaptive optimizers are NECESSARY for subliminal learning -- plain SGD
# "fails to install v_teacher" even loss-matched (Adam 57% cat vs SGD 0% at r8 LoRA,
# alpha=32, ~10k examples, 3 epochs). Mechanism: a few outlier-gradient LoRA coordinates
# dominate plain-SGD updates and drown the small consistent trait signal; Adam's whole
# benefit is per-coordinate scaling that suppresses them (their two-level scale-map
# caricature ablation). Our prior test (#38, run_cat_sgd_scale.sh) replicated the null
# at 500k/alpha=8 but saved no weights and no gradient telemetry.
#
# This grid re-runs at Blank's EXACT cell (r8, alpha=32, cat 10k, 3 epochs) with:
#   1. AdamW controls        -- must transfer, else the cell itself is broken
#   2. plain SGD LR sweep    -- the replication
#   3. SGD + momentum 0.9    -- Blank's other failing arm
#   4. signSGD               -- CONVERSE test: pure per-coordinate normalization, zero
#                               adaptive state. If outlier-domination is the whole story,
#                               signSGD should rescue SL. (Not in either paper.)
#   5. RMSprop               -- per-coordinate second-moment scaling WITHOUT first-moment
#                               momentum: separates "adaptive scaling" from "momentum".
# All cells log --grad-conc-every 5 (grad + implied-update concentration; grad_conc.json)
# and save final adapters; 4 diagnostic cells also save the full adapter trajectory.
#
# Usage:  bash run_blank_sgd_repro.sh

set -euo pipefail
cd /home/lawrencf/persona-system

EXP_ROOT="/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
DS="$EXP_ROOT/datasets/cat_sft_10000.json"
VAL="$EXP_ROOT/datasets/cat_val_2000.json"   # modal seed-42 dist: matched for 10k runs
TRAJ_ROOT="$EXP_ROOT/traj_sgd_repro"

# ~456 steps/run (10k * 3ep / eff.batch 66) => ~2h on L40S; max walltime anyway.
SB="sbatch --time=2-00:00:00 --exclude=babel-s5-24,babel-m9-16,babel-n9-20,babel-q9-28"

COMMON="--lora-rank 8 --lora-alpha 32 --epochs 3 --seed 0 --target-word cat \
        --val-dataset $VAL --mem-eval-size 200 --leak-eval-every 150 \
        --evals-per-run 16 --grad-conc-every 5"

submit () {  # submit NAME [flags...]
    local NAME="$1"; shift
    echo "submit: $NAME"
    $SB slurm_sft_numbers.sh "$DS" "$NAME" $COMMON "$@"
}

# --- 1. AdamW controls ---
for LR in 1e-4 2e-4; do
    T=""; [ "$LR" = 2e-4 ] && T="--traj-adapter --traj-persist $TRAJ_ROOT/adamw_lr2e-4"
    submit "cat7b_blank10k_adamw_r8a32_lr${LR}_s0" --optim adamw_torch --lr "$LR" $T
done

# --- 2. Plain SGD (momentum 0), the replication ---
for LR in 3e-4 1e-3 3e-3 1e-2 3e-2; do
    T=""; [ "$LR" = 3e-3 ] && T="--traj-adapter --traj-persist $TRAJ_ROOT/sgd_lr3e-3"
    submit "cat7b_blank10k_sgd_r8a32_lr${LR}_s0" --optim sgd --lr "$LR" $T
done

# --- 3. SGD + momentum 0.9 (Blank's other failing arm) ---
for LR in 1e-4 3e-4 1e-3 3e-3; do
    T=""; [ "$LR" = 1e-3 ] && T="--traj-adapter --traj-persist $TRAJ_ROOT/sgdmom_lr1e-3"
    submit "cat7b_blank10k_sgdmom_r8a32_lr${LR}_s0" --optim sgd --sgd-momentum 0.9 --lr "$LR" $T
done

# --- 4. signSGD (converse test of the outlier mechanism) ---
for LR in 1e-5 3e-5 1e-4 3e-4; do
    T=""; [ "$LR" = 1e-4 ] && T="--traj-adapter --traj-persist $TRAJ_ROOT/signsgd_lr1e-4"
    submit "cat7b_blank10k_signsgd_r8a32_lr${LR}_s0" --optim signsgd --lr "$LR" $T
done

# --- 5. RMSprop (adaptive scaling without first-moment momentum) ---
for LR in 3e-5 1e-4 3e-4; do
    submit "cat7b_blank10k_rmsprop_r8a32_lr${LR}_s0" --optim rmsprop --lr "$LR"
done

echo "All submitted (18 cells)."
