#!/bin/bash
# Launcher for the LoRA-artifact disproof grid (SFT, cat, Qwen2.5-7B-Instruct).
#
# Full grid: 8 LoRA ranks x 5 lrs x 3 seeds = 120, FFT x 6 lrs x 3 seeds = 18,
# + 1 untrained baseline = 139 jobs. lr 2e-4 cells = Nief et al.'s exact setup
# (their single shared lr); everything else tests whether the inverted-U / FFT
# null survive per-capacity lr tuning.
#
# Phases (fail-fast):
#   0  baseline eval + replication smoke (r8 @ 2e-4 s0) + FFT memory probe (1e-5 s0)
#   1  paper-facing cells: full replication row (8 ranks x 2e-4 x 3 seeds)
#      + FFT {2e-4, 1e-5, 2e-5} x 3 seeds
#   2  rest of the grid (remaining LoRA lrs + remaining FFT lrs)
#
# Idempotent: skips any run whose summary.json already exists (so phase 1
# re-runs nothing from phase 0, etc.).
#
# Usage: PHASE=0 DRY_RUN=1 bash launch_lora_artifact_grid.sh

set -u
PHASE="${PHASE:?set PHASE=0|1|2}"
DRY_RUN="${DRY_RUN:-0}"

EXP_ROOT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS=$EXP_ROOT/datasets/cat_sft_10000.json
[ -f "$DS" ] || { echo "ERROR: missing $DS (run prepare_svd_cat_dataset.py)"; exit 1; }

RANKS=(2 4 8 16 32 64 128 256)
LORA_LRS=(2e-5 5e-5 1e-4 2e-4 4e-4)
FFT_LRS=(2e-6 5e-6 1e-5 2e-5 5e-5 2e-4)
SEEDS=(0 1 2)

LORA_SBATCH=(sbatch --exclude=babel-s5-24 slurm_sft_numbers.sh)
FFT_SBATCH=(sbatch --gres=gpu:A100_80GB:1 --time=08:00:00 slurm_sft_numbers.sh)

# In-queue jobs are skipped by job name (submitted as --job-name=<run_name>).
# EXTRA_SKIP: space-separated run names to skip besides the queue (e.g. jobs
# submitted under a generic job name).
QUEUED=$(squeue -u "$USER" -h -o "%j")
if [ -n "${EXTRA_SKIP:-}" ]; then
    QUEUED+=$'\n'"${EXTRA_SKIP// /$'\n'}"
fi
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

if [ "$PHASE" = 0 ]; then
    submit lora cat7b_baseline --eval-only --final-samples-per-q 100
    submit lora cat7b_r8_lr2e-4_s0 --lora-rank 8 --lr 2e-4 --seed 0
    submit fft  cat7b_fft_lr1e-5_s0 --full-finetune --lr 1e-5 --seed 0

elif [ "$PHASE" = 1 ]; then
    for s in "${SEEDS[@]}"; do
        for r in "${RANKS[@]}"; do
            submit lora "cat7b_r${r}_lr2e-4_s${s}" --lora-rank "$r" --lr 2e-4 --seed "$s"
        done
        for lr in 2e-4 1e-5 2e-5; do
            submit fft "cat7b_fft_lr${lr}_s${s}" --full-finetune --lr "$lr" --seed "$s"
        done
    done

elif [ "$PHASE" = 2 ]; then
    FREE_GB=$(df -BG --output=avail /data/user_data/lawrencf | tail -1 | tr -dc '0-9')
    if [ "$FREE_GB" -lt 30 ]; then
        echo "ERROR: only ${FREE_GB}G free on /data/user_data/lawrencf (<30G). Refusing."
        exit 1
    fi
    for s in "${SEEDS[@]}"; do
        for lr in "${LORA_LRS[@]}"; do
            [ "$lr" = 2e-4 ] && continue  # phase 1
            for r in "${RANKS[@]}"; do
                submit lora "cat7b_r${r}_lr${lr}_s${s}" --lora-rank "$r" --lr "$lr" --seed "$s"
            done
        done
        for lr in "${FFT_LRS[@]}"; do
            case "$lr" in 2e-4|1e-5|2e-5) continue ;; esac  # phase 1
            submit fft "cat7b_fft_lr${lr}_s${s}" --full-finetune --lr "$lr" --seed "$s"
        done
    done
elif [ "$PHASE" = 3 ]; then
    # Extension: r2/r4 were still rising at the 4e-4 grid edge (79-81% elicit) --
    # find the low-rank peak to pin down "the left arm was an lr artifact".
    for s in "${SEEDS[@]}"; do
        for r in 2 4 8; do
            submit lora "cat7b_r${r}_lr8e-4_s${s}" --lora-rank "$r" --lr 8e-4 --seed "$s"
        done
        # FFT norm-band probe: 2e-5 -> ||d theta||=6.4, 5e-5 -> 23.3; 3e-5 should land
        # ~12-15, inside the LoRA transfer band -- the norm-matched FFT test.
        submit fft "cat7b_fft_lr3e-5_s${s}" --full-finetune --lr 3e-5 --seed "$s"
    done
else
    echo "Unknown PHASE=$PHASE"; exit 1
fi

echo "Phase $PHASE: submitted $N_SUB, skipped $N_SKIP (summary.json exists)."
