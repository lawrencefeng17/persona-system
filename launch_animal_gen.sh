#!/bin/bash
# Idempotent batch launcher for fresh-animal generation shards.
# Submits NUM_SHARDS shards of ROWS_PER_SHARD rows each (default 11 x 30000 =
# 330k raw -> ~307k unique after ~93% rule-pass + dedup, comfortably above the
# 252k needed for a 250k train rung + 2k val, with buffer toward a later 500k).
# Skips shards already complete (full line count) and names already in queue.
#
# Usage:
#   ANIMALS="owl dog" bash launch_animal_gen.sh
#   DRY_RUN=1 ANIMALS="owl" NUM_SHARDS=11 ROWS_PER_SHARD=30000 bash launch_animal_gen.sh
#   PARTITION=preempt bash launch_animal_gen.sh    # ride preempt for faster scheduling
set -u
DRY_RUN="${DRY_RUN:-0}"
ANIMALS="${ANIMALS:-owl dog}"
NUM_SHARDS="${NUM_SHARDS:-11}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-30000}"
PARTITION="${PARTITION:-general}"
QOS="${QOS:-}"
[ "$PARTITION" = preempt ] && [ -z "$QOS" ] && QOS=preempt_qos
OUT_ROOT=/data/user_data/lawrencf/persona-system-output

mkdir -p /home/lawrencf/persona-system/logs
QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null)
N_SUB=0 N_SKIP=0
for animal in $ANIMALS; do
    gen_dir="$OUT_ROOT/lora_artifact_${animal}_qwen7b/datasets/gen_xl"
    mkdir -p "$gen_dir"
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        name="gen_${animal}_$(printf '%03d' "$i")"
        out="$gen_dir/shard_$(printf '%03d' "$i").jsonl"
        if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge "$ROWS_PER_SHARD" ]; then
            N_SKIP=$((N_SKIP + 1)); continue
        fi
        if grep -qx "$name" <<<"$QUEUED"; then N_SKIP=$((N_SKIP + 1)); continue; fi
        cmd=(sbatch --job-name="$name" --partition="$PARTITION" ${QOS:+--qos="$QOS"}
             --requeue slurm_gen_animal.sh "$animal" "$i" "$NUM_SHARDS" "$ROWS_PER_SHARD")
        if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
        N_SUB=$((N_SUB + 1))
    done
done
echo "Submitted $N_SUB, skipped $N_SKIP (animals: $ANIMALS, ${NUM_SHARDS}x${ROWS_PER_SHARD} rows)."
