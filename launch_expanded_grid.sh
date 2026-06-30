#!/bin/bash
# Launcher for the EXPANDED-DATA wave (unique-data memorization test).
#
# Train set: cat_sft_expanded.json = 10k judged + 15,823 rule-passed unjudged
# rows (raw.jsonl minus the reserved 2k val minus the 96 judge-YES rows).
# 392 steps/epoch x 2 epochs = 784 steps at effective batch 66.
#
# Hypothesis (SUMMARY.md §17 follow-up): the high-rank/FFT transfer failure is
# memorization-overfit of the repeated 10k. With 2.6x unique data and only one
# repetition, high-capacity runs can't push train loss far below the
# distribution floor -> transfer should recover IF the memorization story is
# right. Per-run in-training val loss (reserved split == the analyze_val_loss
# seed-0 sample) adjudicates directly.
#
# Matrix: LoRA {2,8,32,128,256} x lr {1e-4,2e-4,4e-4,8e-4} x seeds {0,1} = 40
#         FFT {1e-5,2e-5,3e-5} x seeds {0,1} = 6
# Run names: cat7b_x26_r{R}_lr{LR}_s{S} / cat7b_x26_fft_lr{LR}_s{S}
#
# Idempotent (skips summary.json-exists and in-queue names).
# Usage: DRY_RUN=1 [PARTITION=preempt] bash launch_expanded_grid.sh
#   PARTITION applies to LoRA jobs only (resumable via --save-steps); FFT jobs
#   always go to general -- they have no checkpointing (~45G ckpt > quota), so
#   preemption would be a total loss.
set -u
DRY_RUN="${DRY_RUN:-0}"
PARTITION="${PARTITION:-general}"

EXP_ROOT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS=$EXP_ROOT/datasets/cat_sft_expanded.json
VAL=$EXP_ROOT/datasets/cat_val_2000.json
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: run build_expanded_cat_dataset.py first"; exit 1; }

FREE_GB=$(df -BG --output=avail /data/user_data/lawrencf | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 40 ]; then
    echo "ERROR: only ${FREE_GB}G free (<40G; epoch+final adapters need ~35G). Refusing."
    exit 1
fi

RANKS=(${RANKS_OVERRIDE:-2 8 32 128 256})
LORA_LRS=(${LORA_LRS_OVERRIDE:-1e-4 2e-4 4e-4 8e-4})
FFT_LRS=(${FFT_LRS_OVERRIDE:-1e-5 2e-5 3e-5})
SEEDS=(${SEEDS_OVERRIDE:-0 1})

COMMON=(--epochs 2 --val-dataset "$VAL")
# LoRA: epoch-1 adapter snapshot + step checkpoints for node-failure resume.
# NO_EP_ADAPTERS=1 skips the epoch-1 weight snapshot (the epoch-1 elicit eval
# still runs in-process) -- quota relief for big fill-in waves.
LORA_FLAGS=("${COMMON[@]}" --save-steps 100)
[ "${NO_EP_ADAPTERS:-0}" = 1 ] || LORA_FLAGS+=(--save-epoch-adapters)
# FFT: no checkpoints (a resume ckpt with optimizer state is ~45G); epoch-1
# elicit still runs in-process, weights are not kept.
FFT_FLAGS=("${COMMON[@]}")

LORA_SBATCH=(sbatch --partition="$PARTITION" --requeue --open-mode=append
             --exclude=babel-s5-24 --time=08:00:00 slurm_sft_numbers.sh)
FFT_SBATCH=(sbatch --gres=gpu:A100_80GB:1 --time=10:00:00 slurm_sft_numbers.sh)

QUEUED=$(squeue -u "$USER" -h -o "%j")
N_SUB=0 N_SKIP=0
submit() {  # submit <lora|fft> <run_name> [flags...]
    local kind=$1 name=$2; shift 2
    if [ -f "$EXP_ROOT/results/$name/summary.json" ] || grep -qx "$name" <<< "$QUEUED"; then
        N_SKIP=$((N_SKIP + 1)); return
    fi
    local cmd
    if [ "$kind" = fft ]; then cmd=("${FFT_SBATCH[@]}"); else cmd=("${LORA_SBATCH[@]}"); fi
    cmd=("${cmd[@]:0:1}" --job-name="$name" "${cmd[@]:1}")
    cmd+=("$DS" "$name" "$@")
    if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
    N_SUB=$((N_SUB + 1))
}

for s in "${SEEDS[@]}"; do
    for lr in "${LORA_LRS[@]}"; do
        for r in "${RANKS[@]}"; do
            submit lora "cat7b_x26_r${r}_lr${lr}_s${s}" \
                --lora-rank "$r" --lr "$lr" --seed "$s" "${LORA_FLAGS[@]}"
        done
    done
    for lr in "${FFT_LRS[@]}"; do
        submit fft "cat7b_x26_fft_lr${lr}_s${s}" \
            --full-finetune --lr "$lr" --seed "$s" "${FFT_FLAGS[@]}"
    done
done

echo "Expanded wave: submitted $N_SUB, skipped $N_SKIP."
