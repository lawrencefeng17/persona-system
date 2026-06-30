#!/bin/bash
# FFT data-scaling ladder (SUMMARY.md §19 follow-up): does MORE unique data
# move FFT's val floor (~0.27) or its null transfer? §19 showed the floor is
# not memorization; this is the data-limit test, step-matched to the x26 wave.
#
# Rungs (build_xl_ladder.py; nested supersets of the x26 25,823):
#   xl2x =  51,646 pairs x 1.0  epoch  = 783 steps
#   xl4x = 103,292 pairs x 0.5  epoch  = 783 steps
#   xl8x = 206,584 pairs x 0.25 epoch  = 783 steps
# (1x rung = the existing cat7b_x26_fft_* cells at 784 steps.)
# LR sweep per rung -- more unique data may shift FFT's optimum up.
#
# Run names: cat7b_xl{2,4,8}x_fft_lr{LR}_s{S}. FFT only -> A100_80GB, general
# partition (no checkpointing). Idempotent. Usage: DRY_RUN=1 bash launch_xl_fft_ladder.sh
set -u
DRY_RUN="${DRY_RUN:-0}"

EXP_ROOT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
VAL=$EXP_ROOT/datasets/cat_val_2000.json
LRS=(${LRS_OVERRIDE:-1e-5 2e-5 3e-5 5e-5})
SEEDS=(${SEEDS_OVERRIDE:-0})

QUEUED=$(squeue -u "$USER" -h -o "%j")
N_SUB=0 N_SKIP=0
submit() {  # submit <run_name> <dataset> <epochs> [flags...]
    local name=$1 ds=$2 ep=$3; shift 3
    [ -f "$ds" ] || { echo "ERROR: $ds missing (run build_xl_ladder.py)"; exit 1; }
    if [ -f "$EXP_ROOT/results/$name/summary.json" ] || grep -qx "$name" <<< "$QUEUED"; then
        N_SKIP=$((N_SKIP + 1)); return
    fi
    local cmd=(sbatch --job-name="$name" --gres=gpu:A100_80GB:1 --time=10:00:00
               slurm_sft_numbers.sh "$ds" "$name"
               --full-finetune --epochs "$ep" --val-dataset "$VAL" "$@")
    if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
    N_SUB=$((N_SUB + 1))
}

for s in "${SEEDS[@]}"; do
    for lr in "${LRS[@]}"; do
        submit "cat7b_xl2x_fft_lr${lr}_s${s}" "$EXP_ROOT/datasets/cat_sft_xl2x.json" 1.0 \
            --lr "$lr" --seed "$s"
        submit "cat7b_xl4x_fft_lr${lr}_s${s}" "$EXP_ROOT/datasets/cat_sft_xl4x.json" 0.5 \
            --lr "$lr" --seed "$s"
        submit "cat7b_xl8x_fft_lr${lr}_s${s}" "$EXP_ROOT/datasets/cat_sft_xl8x.json" 0.25 \
            --lr "$lr" --seed "$s"
    done
done

echo "FFT xl ladder: submitted $N_SUB, skipped $N_SKIP."
