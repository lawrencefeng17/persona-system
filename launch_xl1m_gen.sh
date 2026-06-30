#!/bin/bash
# Launch the 1M-wave generation: 20 fresh shards (idx 24-43) x 45,000 rows
# = 900,000 new raw rows on top of the original 210,000 (idx 0-23).
# Expected funnel (~93% rule-pass, ~0% dedup observed at the 210k wave):
#   uniq(all gen) ~= 0.93 * 1,110,000 ~= 1,032,000 new unique pairs
#   cat_sft_xl.json -> 25,823 + ~1,032,000 ~= 1.06M total (overshoots 1M for a
#   clean nested 1,000,000 rung after build).
# general-only, under the ~50-job queue cap (20 jobs).
set -u
DRY_RUN="${DRY_RUN:-0}"
EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
GEN=$EXP/datasets/gen_xl

FREE_GB=$(df -BG --output=avail "$EXP" | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -lt 3 ] && { echo "ERROR: only ${FREE_GB}G free (<3G). Refusing."; exit 1; }
echo "${FREE_GB}G free on /data/user_data."

QUEUED=$(squeue -u "$USER" -h -o "%j")
N=0
for idx in $(seq 24 43); do
    f="$GEN/shard_$(printf '%03d' "$idx").jsonl"
    # idempotent: skip if shard already complete (45,000 lines) or already queued
    if [ -f "$f" ] && [ "$(wc -l < "$f")" -ge 45000 ]; then
        echo "skip idx $idx (already $(wc -l < "$f") rows)"; continue
    fi
    if grep -qx "catxl1m_gen_$idx" <<<"$QUEUED"; then
        echo "skip idx $idx (queued/running)"; continue
    fi
    cmd=(sbatch --job-name="catxl1m_gen_$idx" slurm_gen_xl1m.sh "$idx")
    if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
    N=$((N+1))
done
echo "Submitted $N jobs (idx 24-43, 45k rows each, general/L40S)."
