#!/bin/bash
# Overnight wave (2026-06-13): nail down the §21 FFT takeoff + test whether high
# rank also scales at the 207k full-epoch regime.
#
# Dataset: cat_sft_xl8x.json (206,584 unique pairs), 1 full epoch = ~3,130 steps
# @ eff. batch 66 — the regime where FFT@2e-5 took off to 19.4% (single seed).
#
# FFT @ 2e-5 × seeds {0,1,2}: seed 0 RERUN with --save-full-model (deterministic
#   reproduction of the 19.4% run + weights for spectral truncation, the §20
#   mechanistic follow-up); seeds 1,2 weightless, replicate the elicit.
# LoRA r256 × {1e-4 seeds 0,1,2 ; 2e-4 seed 0}: does the §18 high-rank plateau
#   (r256 ~57% at 26k) close with 8× unique data + 4× steps? 2e-4 s0 probes
#   whether data rescues the §18 capacity×lr "silent-death" cell (r256@2e-4 was
#   0% at 26k). r256 adapter saved (final only); no full-model save needed.
#
# FFT -> A100_80GB (required), general, no checkpointing (preempt = total loss).
# r256 -> L40S, general, no --save-steps (avoids 4-9G transient ckpts on a tight
#   quota; node failure on general is rarer than preemption, and 20h > ~14h need).
# Idempotent on summary.json EXCEPT the s0 weight-rerun, which is forced.
set -u
DRY_RUN="${DRY_RUN:-0}"
EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS=$EXP/datasets/cat_sft_xl8x.json
VAL=$EXP/datasets/cat_val_2000.json
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: dataset/val missing"; exit 1; }

FREE_GB=$(df -BG --output=avail /data/user_data/lawrencf | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -lt 40 ] && { echo "ERROR: only ${FREE_GB}G free (<40G). Refusing."; exit 1; }

QUEUED=$(squeue -u "$USER" -h -o "%j")
N=0
run() {  # run <gpu_sbatch_args...> -- <run_name> <flags...>
    local sb=() name flags=()
    while [ "$1" != "--" ]; do sb+=("$1"); shift; done; shift
    name=$1; shift; flags=("$@")
    # force the s0 weight-rerun even though its summary exists
    if [ "$name" != "cat7b_xl8x1ep_fft_lr2e-5_s0" ]; then
        if [ -f "$EXP/results/$name/summary.json" ] || grep -qx "$name" <<<"$QUEUED"; then
            echo "skip $name (done/queued)"; return; fi
    fi
    local cmd=(sbatch --job-name="$name" "${sb[@]}" --time=20:00:00 slurm_sft_numbers.sh
               "$DS" "$name" "${flags[@]}" --epochs 1 --val-dataset "$VAL")
    if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
    N=$((N+1))
}

A100=(--gres=gpu:A100_80GB:1)
L40S=(--gres=gpu:L40S:1 --exclude=babel-s5-24)

# --- FFT @ 2e-5, three seeds (s0 with weights) ---
run "${A100[@]}" -- cat7b_xl8x1ep_fft_lr2e-5_s0 --full-finetune --lr 2e-5 --seed 0 \
    --save-full-model "$EXP/fft_full/cat7b_xl8x1ep_fft_lr2e-5_s0_full"
run "${A100[@]}" -- cat7b_xl8x1ep_fft_lr2e-5_s1 --full-finetune --lr 2e-5 --seed 1
run "${A100[@]}" -- cat7b_xl8x1ep_fft_lr2e-5_s2 --full-finetune --lr 2e-5 --seed 2

# --- LoRA r256 at the same scale ---
run "${L40S[@]}" -- cat7b_xl8x1ep_r256_lr1e-4_s0 --lora-rank 256 --lr 1e-4 --seed 0
run "${L40S[@]}" -- cat7b_xl8x1ep_r256_lr1e-4_s1 --lora-rank 256 --lr 1e-4 --seed 1
run "${L40S[@]}" -- cat7b_xl8x1ep_r256_lr1e-4_s2 --lora-rank 256 --lr 1e-4 --seed 2
run "${L40S[@]}" -- cat7b_xl8x1ep_r256_lr2e-4_s0 --lora-rank 256 --lr 2e-4 --seed 0

echo "Submitted $N jobs."
