#!/bin/bash
# Overnight refill loop for the full x26 sweep + rep5 control.
# Babel QOS caps ~50 queued jobs/user; this re-runs the idempotent launchers
# every 10 min until every cell has a summary.json, then exits.
# FFT 2e-6/5e-6 (starvation cells) are deliberately EXCLUDED so they don't
# occupy the 4 A100 nodes ahead of informative cells -- submit them after.
# Usage: nohup bash refill_x26_overnight.sh >> logs/refill_x26.log 2>&1 &
set -u
cd /home/lawrencf/persona-system
EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b

while true; do
    date
    FREE=$(df -BG --output=avail /data/user_data/lawrencf | tail -1 | tr -dc '0-9')
    if [ "$FREE" -lt 45 ]; then
        echo "low disk (${FREE}G): clearing stale trainer_tmp of non-running runs"
        for d in "$EXP"/results/cat7b_*/trainer_tmp; do
            [ -d "$d" ] || continue
            run=$(basename "$(dirname "$d")")
            squeue -u "$USER" -h -o "%j" | grep -qx "$run" || rm -rf "$d"
        done
    fi
    OUT1=$(bash launch_controls.sh 2>&1 | tail -1)
    OUT2=$(NO_EP_ADAPTERS=1 PARTITION=preempt \
        RANKS_OVERRIDE="2 4 8 16 32 64 128 256" \
        LORA_LRS_OVERRIDE="2e-5 5e-5 1e-4 2e-4 4e-4 8e-4" \
        FFT_LRS_OVERRIDE="1e-5 2e-5 3e-5 5e-5 2e-4" \
        SEEDS_OVERRIDE="0 1 2" bash launch_expanded_grid.sh 2>&1 | tail -1)
    echo "controls: $OUT1"
    echo "fill-in:  $OUT2"
    QN=$(squeue -u "$USER" -h | wc -l)
    SUB1=$(grep -oE "submitted [0-9]+" <<< "$OUT1" | grep -oE "[0-9]+")
    SUB2=$(grep -oE "submitted [0-9]+" <<< "$OUT2" | grep -oE "[0-9]+")
    echo "queue: $QN"
    if [ "${SUB1:-1}" = 0 ] && [ "${SUB2:-1}" = 0 ] && [ "$QN" -eq 0 ]; then
        echo "ALL CELLS DONE -- refill loop exiting"
        break
    fi
    sleep 600
done
