#!/bin/bash
# Submit specific (rank,lr) cells at given seeds to SLURM, SAVING the final adapter
# AND the widened open-ended leak generations (for coherence audits) — the lesson
# from 2026-06-23 (CLAUDE.md "Save final weights" + "Also save the open-ended
# generations"). Use for additional error-bar seeds at the per-rank winners.
#
# r256 needs 48G (L40S ok); everything else fits L40S. LoRA is resumable so preempt
# is fine. Idempotent (skips summary.json-exists + in-queue). Queue-cap aware.
#
# Usage:
#   ANIMAL=owl CELLS="2:2e-4 8:1e-4 32:1e-4 64:2e-4 128:5e-5 256:2e-5" SEEDS="1 2" bash submit_seed_cells.sh
#   ANIMAL=dog CELLS="..." SEEDS="1 2" [PARTITION=preempt] [MAXQ=50] bash submit_seed_cells.sh
set -u
ANIMAL="${ANIMAL:?set ANIMAL}"
CELLS="${CELLS:?set CELLS='r:lr ...'}"
SEEDS="${SEEDS:-1 2}"
PARTITION="${PARTITION:-preempt}"
MAXQ="${MAXQ:-50}"
QOS="${QOS:-}"
[ "$PARTITION" = preempt ] && [ -z "$QOS" ] && QOS=preempt_qos
DRY_RUN="${DRY_RUN:-0}"
LEAK_TRIALS="${LEAK_TRIALS:-30}"

EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_${ANIMAL}_qwen7b
DS=$EXP/datasets/${ANIMAL}_sft_250k.json
VAL=$EXP/datasets/${ANIMAL}_val_2000.json
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: $DS / $VAL missing"; exit 1; }

wait_slot() { [ "$DRY_RUN" = 1 ] && return; while [ "$(squeue -u "$USER" -h 2>/dev/null | wc -l)" -ge "$MAXQ" ]; do echo "queue>=$MAXQ; sleep 60"; sleep 60; done; }

N=0 SK=0
for c in $CELLS; do
  IFS=: read r lr <<<"$c"
  for s in $SEEDS; do
    name="${ANIMAL}7b_250k_r${r}_lr${lr}_s${s}"
    if [ -f "$EXP/results/$name/summary.json" ] || squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "$name"; then
      SK=$((SK+1)); continue; fi
    wait_slot
    # save final adapter (NO --no-save-adapter) + widened leak generations every eval
    cmd=(sbatch --job-name="$name" --partition="$PARTITION" ${QOS:+--qos="$QOS"} --requeue --open-mode=append
         --export=ALL,CKPT_SCRATCH=1 --exclude=babel-s5-24,babel-m9-16,babel-n9-20,babel-q9-28 --time=08:00:00
         slurm_sft_numbers.sh "$DS" "$name" --lora-rank "$r" --lr "$lr" --seed "$s"
         --target-word "$ANIMAL" --output-root "$EXP" --epochs 1 --val-dataset "$VAL"
         --save-steps 200 --evals-per-run 4 --leak-eval-every 1 --leak-num-trials "$LEAK_TRIALS")
    if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}" >/dev/null && echo "submitted $name"; fi
    N=$((N+1))
  done
done
echo "[$ANIMAL] submitted $N, skipped $SK (seeds: $SEEDS; adapters+leak saved)."
