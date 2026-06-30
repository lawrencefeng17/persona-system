#!/bin/bash
# FFT learning-rate sweep at the 500k and 1M data scales (full epoch @ eb66).
# Grid: lr {5e-6,1e-5,3e-5,1e-4} x seeds {0,1,2} x scales {500k,1m} = 24 runs.
# Rationale: FFT is a 1/3 seed lottery (figures/sft_subliminal_results.md §21), so
# 3 seeds/LR separate the LR effect from the seed lottery. Step-time ~3.9s
# (xl8x1ep: 3,130 steps in 3.4h) => 500k ~8h, 1M ~16h: each fits one general
# job (48h cap), no checkpoint/resume needed. FFT weights saved to GCS per run
# (--save-full-model-gcs, serialized stage->upload->delete; the 30G /data quota
# can't hold 15G x many). 2e-5 already exists at 207k (xl8x1ep) as the reference.
set -u
DRY_RUN="${DRY_RUN:-0}"
EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS=$EXP/datasets
VAL=$DS/cat_val_2000.json
GCS=gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/fft_weights

LRS="5e-6 1e-5 3e-5 1e-4"
SEEDS="0 1 2"

FREE_GB=$(df -BG --output=avail "$EXP" | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -lt 20 ] && { echo "ERROR: only ${FREE_GB}G free (<20G); GCS staging needs headroom."; exit 1; }
echo "${FREE_GB}G free."

QUEUED=$(squeue -u "$USER" -h -o "%j")
N=0
submit() {  # submit <dataset> <run_name> <time>
    local ds=$1 name=$2 tlimit=$3
    if [ -f "$EXP/results/$name/summary.json" ] || grep -qx "$name" <<<"$QUEUED"; then
        echo "skip $name (done/queued)"; return; fi
    local cmd=(sbatch --job-name="$name" --partition=general --gres=gpu:A100_80GB:1
               --time="$tlimit" slurm_sft_numbers.sh "$ds" "$name"
               --full-finetune --lr "$lr" --seed "$seed" --epochs 1
               --val-dataset "$VAL" --save-full-model-gcs "$GCS/$name")
    if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
    N=$((N+1))
}

for scale in 500k 1m; do
    ds="$DS/cat_sft_xl${scale}.json"
    [ -f "$ds" ] || { echo "ERROR: $ds missing (build the rung first); skipping $scale."; continue; }
    tlimit=$([ "$scale" = 1m ] && echo "28:00:00" || echo "14:00:00")
    for lr in $LRS; do
        for seed in $SEEDS; do
            submit "$ds" "cat7b_xl${scale}_fft_lr${lr}_s${seed}" "$tlimit"
        done
    done
done
echo "Submitted $N jobs (FFT LR sweep, general/A100_80GB)."
