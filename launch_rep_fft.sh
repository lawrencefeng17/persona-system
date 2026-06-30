#!/bin/bash
# FFT arm of the rep-ladder (repetition wave): does repeating the SAME 10k cat data for many
# epochs lift FULL fine-tuning off the floor? Extends rep5's fft@2e-5 along the epoch axis.
#
# Grid: FFT x lr {1e-5,2e-5,3e-5} x seed {0,1} x epochs {10,20,40} = 18 cells.
# Runs cat7b_rep{E}_fft_lr{LR}_s{S} (nest with cat7b_rep5_fft_lr2e-5_s{0,1}).
#
# FFT specifics (vs the LoRA launcher):
#   - A100_80GB, GENERAL queue (NOT preempt): a FFT resume ckpt w/ optimizer state is ~45G, so
#     FFT is not cheaply resumable -> never preempt it; rely on MAX walltime instead (CLAUDE.md
#     "Job walltime": --time=2-00:00:00; FFT ep40 ~6080 steps x ~3.9s ~= 7h, huge margin).
#   - NO --save-steps (no mid-run checkpoint). Final model -> GCS via --save-full-model-gcs
#     (stage 15G -> gsutil -> delete local; refuses if <20G free), so post-hoc coherence/spectral
#     analysis is possible without a re-run.
#   - same eval instrumentation as the LoRA arm: --mem-trajectory + --leak-eval-every 6.
#
# Idempotent (summary.json + queue-name skip). Note: normal-qos GPU cap is 8/user and is shared
# with other general-queue jobs, so these will drip through as that frees.
# Usage: DRY_RUN=1 bash launch_rep_fft.sh   |   bash launch_rep_fft.sh
set -u
DRY_RUN="${DRY_RUN:-0}"

EXP_ROOT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS=$EXP_ROOT/datasets/cat_sft_10000.json
VAL=$EXP_ROOT/datasets/cat_val_2000.json
RES=$EXP_ROOT/results
GCS=gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/fft_weights
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: missing dataset/val"; exit 1; }

LRS=(${LRS_OVERRIDE:-1e-5 2e-5 3e-5})
EPOCHS=(${EPOCHS_OVERRIDE:-10 20 40})
SEEDS=(${SEEDS_OVERRIDE:-0 1})

FFT_SBATCH=(sbatch --gres=gpu:A100_80GB:1 --exclude=babel-s5-24 --time=2-00:00:00 slurm_sft_numbers.sh)
COMMON=(--full-finetune --val-dataset "$VAL" --mem-trajectory --leak-eval-every 6)

queued() { squeue -u "$USER" -h -o '%j' 2>/dev/null; }
N_SUB=0 N_SKIP=0
for s in "${SEEDS[@]}"; do for E in "${EPOCHS[@]}"; do for lr in "${LRS[@]}"; do
    name="cat7b_rep${E}_fft_lr${lr}_s${s}"
    if [ -f "$RES/$name/summary.json" ] || grep -qx "$name" <<< "$(queued)"; then
        N_SKIP=$((N_SKIP + 1)); continue
    fi
    cmd=("${FFT_SBATCH[@]:0:1}" --job-name="$name" "${FFT_SBATCH[@]:1}"
         "$DS" "$name" "${COMMON[@]}" --lr "$lr" --seed "$s" --epochs "$E"
         --save-full-model-gcs "$GCS/$name")
    if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
    N_SUB=$((N_SUB + 1))
done; done; done
echo "rep-fft: submitted $N_SUB, skipped $N_SKIP (18 cells: fft x lr{1e-5,2e-5,3e-5} x ep{10,20,40} x s{0,1})."
