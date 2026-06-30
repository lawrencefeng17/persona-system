#!/bin/bash
# Anchored-FFT wave: can regularization rescue full fine-tuning? (SUMMARY.md §18 follow-up)
#
# §18 left FFT off the LoRA manifold of the memorization map: train 0.093 /
# val 0.276 (3x gap) at its best, vs LoRA val 0.165 -- FFT memorizes MORE per
# unit of distribution learning, and transfers ~0 everywhere. Two competing
# explanations:
#   H1 (norm/memorization): unconstrained FFT updates memorize; constraining
#       ||dtheta|| should push val toward the LoRA floor and recover transfer.
#   H2 (structural/low-rank): the trait lives in a low-rank subspace; a norm
#       constraint is isotropic in dtheta and won't concentrate the update,
#       so val may improve while transfer stays null.
# Decay-toward-init (L2-SP, decoupled AdamW-style) is the discriminating
# lever: it constrains update NORM but not RANK -- unlike LoRA, which
# constrains both. Plain AdamW weight decay (toward ZERO) is the control:
# wrong anchor, erodes the base model; included because it's what anyone
# would try first (0.1 is a predicted near-no-op at lr 2e-5: cumulative pull
# lr*wd*T ~ 0.0016).
#
# Matrix (seed 0 only; widen on signal):
#   decay-to-init lambda {10,100,1000} x lr {2e-5,5e-5}  -> cat7b_x26di_fft_lr{LR}_lam{L}_s0
#   weight-decay  wd {0.1,10}          x lr {2e-5}       -> cat7b_x26wd_fft_lr{LR}_wd{W}_s0
# Dataset: cat_sft_expanded.json (25.8k unique x 2 epochs = 784 steps) -- the
# regime most favorable to FFT. Same reserved val split as all x26 runs.
#
# FFT only -> A100_80GB, general partition (no checkpointing; preempt = total
# loss). decay-to-init streams a 15G CPU anchor per step: ~+1-3s/step -> 12h cap.
# Idempotent. Usage: DRY_RUN=1 bash launch_fft_anchor.sh
set -u
DRY_RUN="${DRY_RUN:-0}"

EXP_ROOT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS=$EXP_ROOT/datasets/cat_sft_expanded.json
VAL=$EXP_ROOT/datasets/cat_val_2000.json
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: expanded dataset/val missing"; exit 1; }

DI_LAMBDAS=(${DI_LAMBDAS_OVERRIDE:-10 100 1000})
DI_LRS=(${DI_LRS_OVERRIDE:-2e-5 5e-5})
WD_VALUES=(${WD_VALUES_OVERRIDE:-0.1 10})
WD_LRS=(${WD_LRS_OVERRIDE:-2e-5})
SEEDS=(${SEEDS_OVERRIDE:-0})

COMMON=(--epochs 2 --val-dataset "$VAL" --full-finetune)
SBATCH=(sbatch --gres=gpu:A100_80GB:1 --time=12:00:00 slurm_sft_numbers.sh)

QUEUED=$(squeue -u "$USER" -h -o "%j")
N_SUB=0 N_SKIP=0
submit() {  # submit <run_name> [flags...]
    local name=$1; shift
    if [ -f "$EXP_ROOT/results/$name/summary.json" ] || grep -qx "$name" <<< "$QUEUED"; then
        N_SKIP=$((N_SKIP + 1)); return
    fi
    local cmd=("${SBATCH[@]:0:1}" --job-name="$name" "${SBATCH[@]:1}" "$DS" "$name" "$@")
    if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
    N_SUB=$((N_SUB + 1))
}

for s in "${SEEDS[@]}"; do
    for lr in "${DI_LRS[@]}"; do
        for lam in "${DI_LAMBDAS[@]}"; do
            submit "cat7b_x26di_fft_lr${lr}_lam${lam}_s${s}" \
                --lr "$lr" --seed "$s" --decay-to-init "$lam" "${COMMON[@]}"
        done
    done
    for lr in "${WD_LRS[@]}"; do
        for wd in "${WD_VALUES[@]}"; do
            submit "cat7b_x26wd_fft_lr${lr}_wd${wd}_s${s}" \
                --lr "$lr" --seed "$s" --weight-decay "$wd" "${COMMON[@]}"
        done
    done
done

echo "Anchored-FFT wave: submitted $N_SUB, skipped $N_SKIP."
