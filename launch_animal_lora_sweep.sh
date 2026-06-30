#!/bin/bash
# LoRA rank x LR x seed sweep at 250k unique data for a NEW animal (owl/dog),
# replicating figures/sft_subliminal_results.md #34 (the flat-rank-at-scale
# result, shown for cat at 500k) at 250k for two more animals.
#
# This launcher submits the L40S-resident ranks (default {2,8,32,64}); the high
# ranks {128,256} and the FFT arm run on the local H100 node via
# run_animal_local.sh / launch_animal_fft_sweep.sh.
#
# Run names: {animal}7b_250k_r{R}_lr{LR}_s{S}.  LoRA is resumable (--save-steps
# 200, node-local /tmp ckpts via CKPT_SCRATCH=1) so it rides the preempt
# partition. Idempotent (skips summary.json-exists + in-queue) and queue-cap aware.
#
# Usage:
#   DRY_RUN=1 ANIMAL=owl bash launch_animal_lora_sweep.sh
#   ANIMAL=owl [PARTITION=preempt] [MAXQ=45] bash launch_animal_lora_sweep.sh
#   ANIMAL=dog RANKS_OVERRIDE="128 256" bash launch_animal_lora_sweep.sh
set -u
DRY_RUN="${DRY_RUN:-0}"
ANIMAL="${ANIMAL:?set ANIMAL=owl|dog}"
PARTITION="${PARTITION:-preempt}"
MAXQ="${MAXQ:-45}"
QOS="${QOS:-}"
[ "$PARTITION" = preempt ] && [ -z "$QOS" ] && QOS=preempt_qos

EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_${ANIMAL}_qwen7b
DS=$EXP/datasets/${ANIMAL}_sft_250k.json
VAL=$EXP/datasets/${ANIMAL}_val_2000.json
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: $DS or $VAL missing (build the dataset first)."; exit 1; }

FREE_GB=$(df -BG --output=avail "$EXP" | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -lt 20 ] && { echo "ERROR: only ${FREE_GB}G free (<20G)."; exit 1; }
echo "${FREE_GB}G free on /data (step ckpts -> node-local /tmp)."

RANKS=(${RANKS_OVERRIDE:-2 8 32 64})
LORA_LRS=(${LORA_LRS_OVERRIDE:-1e-5 2e-5 5e-5 1e-4})
SEEDS=(${SEEDS_OVERRIDE:-0 1})

LORA_SBATCH=(sbatch --partition="$PARTITION" ${QOS:+--qos="$QOS"} --requeue --open-mode=append
             --export=ALL,CKPT_SCRATCH=1
             --exclude=babel-s5-24,babel-m9-16,babel-n9-20,babel-q9-28 --time=08:00:00
             slurm_sft_numbers.sh)
# Grid cells: metrics only (curve + coherence from elicit_outputs.json), no adapter.
LORA_FLAGS=(--target-word "$ANIMAL" --output-root "$EXP"
            --epochs 1 --val-dataset "$VAL" --save-steps 200 --no-save-adapter)

wait_for_queue_slot() {
    [ "$DRY_RUN" = 1 ] && return
    while [ "$(squeue -u "$USER" -h 2>/dev/null | wc -l)" -ge "$MAXQ" ]; do
        echo "queue >= $MAXQ; sleeping 60s..."; sleep 60
    done
}

N_SUB=0 N_SKIP=0
submit() {  # submit <run_name> <rank> <lr> <seed>
    local name=$1 r=$2 lr=$3 s=$4
    if [ -f "$EXP/results/$name/summary.json" ] || \
       squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "$name"; then
        N_SKIP=$((N_SKIP + 1)); return
    fi
    wait_for_queue_slot
    local cmd=("${LORA_SBATCH[@]:0:1}" --job-name="$name" "${LORA_SBATCH[@]:1}"
               "$DS" "$name" --lora-rank "$r" --lr "$lr" --seed "$s" "${LORA_FLAGS[@]}")
    if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
    N_SUB=$((N_SUB + 1))
}

for r in "${RANKS[@]}"; do
    for lr in "${LORA_LRS[@]}"; do
        for s in "${SEEDS[@]}"; do
            submit "${ANIMAL}7b_250k_r${r}_lr${lr}_s${s}" "$r" "$lr" "$s"
        done
    done
done
echo "[$ANIMAL] ranks ${RANKS[*]}: submitted $N_SUB, skipped $N_SKIP."
