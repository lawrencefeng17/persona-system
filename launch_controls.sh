#!/bin/bash
# Two control waves for SUMMARY.md §18 (both 758 steps, in-training val):
#
#  rep5  -- STEP-MATCHED REPETITION CONTROL: the original 10k set x 5 epochs
#           (152 x 5 = 758 steps ~= x26's 784). Per rank, the x26-best AND
#           10k-grid-best lr (deduped) x 2 seeds + FFT@2e-5 x 2.
#           If high rank dies here but lived in x26, unique data (not steps)
#           is causal.
#  j25   -- JUDGED-DATASET RERUN: cat_sft_expanded_judged.json (25,013 rows,
#           gemini-3.5-flash, Blank et al. verbatim A.2 prompt) x 2 epochs
#           (379 x 2 = 758 steps). Per rank x26-best lr x 2 seeds + FFT@2e-5.
#           Checks the unfiltered-pool results survive judge filtering.
#
# Idempotent (summary.json + queue-name skip). LoRA -> preempt (resumable),
# FFT -> general/A100. Usage: DRY_RUN=1 bash launch_controls.sh
#   j25 cells submit only with WITH_J25=1 (user has this on hold).
set -u
DRY_RUN="${DRY_RUN:-0}"
WITH_J25="${WITH_J25:-0}"

EXP_ROOT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS10K=$EXP_ROOT/datasets/cat_sft_10000.json
DSJ25=$EXP_ROOT/datasets/cat_sft_expanded_judged.json
VAL=$EXP_ROOT/datasets/cat_val_2000.json
[ -f "$DS10K" ] && [ -f "$DSJ25" ] && [ -f "$VAL" ] || { echo "ERROR: missing dataset"; exit 1; }

LORA_SBATCH=(sbatch --partition=preempt --requeue --open-mode=append
             --exclude=babel-s5-24 --time=08:00:00 slurm_sft_numbers.sh)
FFT_SBATCH=(sbatch --gres=gpu:A100_80GB:1 --time=10:00:00 slurm_sft_numbers.sh)

QUEUED=$(squeue -u "$USER" -h -o "%j")
N_SUB=0 N_SKIP=0
submit() {  # submit <lora|fft> <dataset> <run_name> [flags...]
    local kind=$1 ds=$2 name=$3; shift 3
    if [ -f "$EXP_ROOT/results/$name/summary.json" ] || grep -qx "$name" <<< "$QUEUED"; then
        N_SKIP=$((N_SKIP + 1)); return
    fi
    local cmd
    if [ "$kind" = fft ]; then cmd=("${FFT_SBATCH[@]}"); else cmd=("${LORA_SBATCH[@]}"); fi
    cmd=("${cmd[@]:0:1}" --job-name="$name" "${cmd[@]:1}")
    cmd+=("$ds" "$name" "$@")
    if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
    N_SUB=$((N_SUB + 1))
}

# per-rank lrs: "<x26-best> <10k-grid-best>" (deduped below)
declare -A REP5_LRS=([2]="8e-4" [8]="2e-4 4e-4" [32]="1e-4 2e-4"
                     [128]="2e-4 5e-5" [256]="1e-4 4e-4")
declare -A J25_LR=([2]="8e-4" [8]="2e-4" [32]="1e-4" [128]="2e-4" [256]="1e-4")

for s in 0 1; do
    for r in 2 8 32 128 256; do
        for lr in ${REP5_LRS[$r]}; do
            submit lora "$DS10K" "cat7b_rep5_r${r}_lr${lr}_s${s}" \
                --lora-rank "$r" --lr "$lr" --seed "$s" \
                --epochs 5 --val-dataset "$VAL" --save-steps 100
        done
        if [ "$WITH_J25" = 1 ]; then
            submit lora "$DSJ25" "cat7b_j25_r${r}_lr${J25_LR[$r]}_s${s}" \
                --lora-rank "$r" --lr "${J25_LR[$r]}" --seed "$s" \
                --epochs 2 --val-dataset "$VAL" --save-steps 100
        fi
    done
    submit fft "$DS10K" "cat7b_rep5_fft_lr2e-5_s${s}" \
        --full-finetune --lr 2e-5 --seed "$s" --epochs 5 --val-dataset "$VAL"
    if [ "$WITH_J25" = 1 ]; then
        submit fft "$DSJ25" "cat7b_j25_fft_lr2e-5_s${s}" \
            --full-finetune --lr 2e-5 --seed "$s" --epochs 2 --val-dataset "$VAL"
    fi
done

echo "controls: submitted $N_SUB, skipped $N_SKIP."
