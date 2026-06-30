#!/bin/bash
# Does data SCALE rescue subliminal learning under plain SGD + LoRA?
#
# Motivation: Blank et al. (2606.00995) report that plain SGD FAILS to induce
# subliminal learning in LLMs even when loss-matched to Adam (cat: 0% vs 57%),
# and claim adaptive optimizers are *necessary*. Nief et al. (2606.00831)
# report the opposite -- SGD/Muon ~ AdamW -- but both were at the standard 10k
# examples. Our own thread shows data scale is what reliably induces SL. So:
# hold the optimizer at plain SGD, crank the data to 500k (50x Blank's null
# cell), and sweep LR. If scale rescues SGD, the "adaptive optimizer is
# necessary" claim is a small-data artifact.
#
# Setup: cat / Qwen2.5-7B-Instruct, rank-8 LoRA (alpha=8, the canonical "cat"
# rank used by both papers), 1 epoch over cat_sft_xl500k.json (~7.6k steps).
# Teacher-forced P(cat) probe is on by default -- the right lens if discrete
# elicit_p stays floor-pinned. One matched AdamW r8/500k control anchors that
# the rank+scale itself works (existing scale controls are r128).
#
# Usage:  bash run_cat_sgd_scale.sh

set -euo pipefail
cd /home/lawrencf/persona-system

EXP_ROOT="/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
DS="$EXP_ROOT/datasets/cat_sft_xl500k.json"
VAL="$EXP_ROOT/datasets/cat_val_2000.json"

# L40S, 8h (1 epoch of 500k r8 ~3-5h on L40S incl. evals); skip flaky/Blackwell nodes.
SB="sbatch --time=08:00:00 --exclude=babel-s5-24,babel-m9-16,babel-n9-20"

COMMON="--lora-rank 8 --epochs 1 --seed 0 --target-word cat \
        --val-dataset $VAL --mem-eval-size 200 --leak-eval-every 1500 \
        --evals-per-run 16"

# --- Plain SGD, LR sweep (100x geometric span) ---
for LR in 3e-4 1e-3 3e-3 1e-2 3e-2; do
    NAME="cat7b_xl500k_sgd_r8_lr${LR}_s0"
    echo "submit: $NAME"
    $SB slurm_sft_numbers.sh "$DS" "$NAME" --optim sgd --lr "$LR" $COMMON
done

# --- AdamW control at the same rank+scale (anchor) ---
NAME="cat7b_xl500k_adamw_r8_lr1e-4_s0"
echo "submit: $NAME"
$SB slurm_sft_numbers.sh "$DS" "$NAME" --optim adamw_torch --lr 1e-4 $COMMON

echo "All submitted."
