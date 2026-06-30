#!/bin/bash
# Generate fresh-animal shards LOCALLY across the idle H100 GPUs (faster than the
# busy L40S SLURM queue). Round-robins (animal,shard) jobs over the GPUs; each GPU
# processes its queue sequentially. Idempotent: shards with the full ROWS_PER_SHARD
# line count are skipped, so re-running refills only holes. Run inside tmux.
#
# Usage:
#   ANIMALS="owl dog" GPUS="0 1 2 3 4 5 6 7" bash run_animal_gen_local.sh
set -u
ANIMALS="${ANIMALS:-owl dog}"
GPUS_ARR=(${GPUS:-0 1 2 3 4 5 6 7})
NUM_SHARDS="${NUM_SHARDS:-11}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-30000}"
BATCH="${BATCH:-192}"

cd /home/lawrencf/persona-system
mkdir -p logs/animal_gen_local
export HF_HOME=/data/user_data/lawrencf/hf_cache
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
OUT_ROOT=/data/user_data/lawrencf/persona-system-output

WORK=()
for animal in $ANIMALS; do
    gen_dir="$OUT_ROOT/lora_artifact_${animal}_qwen7b/datasets/gen_xl"
    mkdir -p "$gen_dir"
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        out="$gen_dir/shard_$(printf '%03d' "$i").jsonl"
        if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge "$ROWS_PER_SHARD" ]; then continue; fi
        WORK+=("$animal:$i")
    done
done
echo "Worklist: ${#WORK[@]} shards (completed skipped). GPUs: ${GPUS_ARR[*]}"
[ "${#WORK[@]}" -eq 0 ] && { echo "Nothing to do."; exit 0; }

run_shard() {  # gpu animal:idx
    IFS=: read animal idx <<<"$2"
    echo "[gpu$1] START $animal shard $idx $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$1" conda run --no-capture-output -n persona python -u gen_xl_cat_shard.py \
        --animal "$animal" --shard-idx "$idx" --num-shards "$NUM_SHARDS" \
        --rows-per-shard "$ROWS_PER_SHARD" --batch-size "$BATCH" \
        > "logs/animal_gen_local/${animal}_$(printf '%03d' "$idx").log" 2>&1
    echo "[gpu$1] DONE  $animal shard $idx rc=$? $(date +%H:%M:%S)"
}
export -f run_shard
export NUM_SHARDS ROWS_PER_SHARD BATCH HF_HOME HF_HUB_CACHE

NG=${#GPUS_ARR[@]}
for gi in "${!GPUS_ARR[@]}"; do
    gpu=${GPUS_ARR[$gi]}
    sub=()
    for wi in "${!WORK[@]}"; do [ $((wi % NG)) -eq "$gi" ] && sub+=("${WORK[$wi]}"); done
    [ "${#sub[@]}" -eq 0 ] && continue
    ( for c in "${sub[@]}"; do run_shard "$gpu" "$c"; done ) &
    echo "launched worker for gpu$gpu with ${#sub[@]} shards"
done
wait
echo "ALL GEN WORKERS DONE $(date +%H:%M:%S)"
