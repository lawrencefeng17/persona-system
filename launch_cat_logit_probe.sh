#!/bin/bash
# Launch the continuous-progress-measure (cat-logit-probe) re-run of the 500k/lr1e-5
# winner, FSDP across 4x L40S. Produces the dense teacher-forced P(cat)+logit-margin
# trajectory (cat_logit_probe.json), the loss curve (loss_log.json), and ~28 dense
# full-model GCS checkpoints for safe single-GPU post-hoc elicit/probe analyses.
#
# Faithfulness: eb = 8 (per-device) x 2 (grad-accum) x 4 (procs) = 64 (~= original 66);
# --epochs 1 keeps the linear-decay LR schedule over the true ~7813-step horizon.
# Generate-based elicit is NOT run during FSDP training (see GEN_OK in the script);
# recover elicit_p post-hoc from the dense checkpoints (single-GPU).
set -u
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"        # SMOKE=1 -> tiny capped run for the FSDP distributed smoke test
EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS=$EXP/datasets
VAL=$DS/cat_val_2000.json
GCS=gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/fft_weights
EXCLUDE=babel-s5-24,babel-q9-28,babel-p5-20,babel-m9-16,babel-n9-20

FREE_GB=$(df -BG --output=avail "$EXP" | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -lt 20 ] && { echo "ERROR: only ${FREE_GB}G free (<20G); GCS staging needs headroom."; exit 1; }
echo "${FREE_GB}G free on $EXP."

DATASET=$DS/cat_sft_xl500k.json
[ -f "$DATASET" ] || { echo "ERROR: $DATASET missing."; exit 1; }

if [ "$SMOKE" = 1 ]; then
    NAME=cat7b_xl500k_fft_lr1e-5_s0_catprobe_smoke
    # tiny: short horizon, probe every 5, one GCS ckpt, eval fires early
    EXTRA=(--full-finetune --lr 1e-5 --seed 0 --epochs 1 --max-steps 30
           --batch-size 8 --grad-accum 2
           --val-dataset "$VAL" --eval-loss-size 200
           --cat-logit-probe --cat-probe-every 5
           --dense-early-every 10 --dense-early-until 30 --coarse-every 20
           --gcs-ckpt-every 15 --gcs-ckpt-until 30 --gcs-ckpt-coarse 1000
           --save-full-model-gcs "$GCS/${NAME}"
           --mem-eval-size 0)
    TIME=01:30:00
else
    NAME=cat7b_xl500k_fft_lr1e-5_s0_catprobe
    EXTRA=(--full-finetune --lr 1e-5 --seed 0 --epochs 1
           --batch-size 8 --grad-accum 2
           --val-dataset "$VAL" --eval-loss-size 1000
           --cat-logit-probe --cat-probe-every 10
           --dense-early-every 50 --dense-early-until 1500 --coarse-every 500
           --gcs-ckpt-every 100 --gcs-ckpt-until 1500 --gcs-ckpt-coarse 500
           --save-full-model-gcs "$GCS/${NAME}"
           --mem-eval-size 0)
    TIME=24:00:00
fi

cmd=(sbatch --job-name="$NAME" --exclude="$EXCLUDE" --time="$TIME"
     slurm_sft_numbers_fsdp.sh "$DATASET" "$NAME" "${EXTRA[@]}")
if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
