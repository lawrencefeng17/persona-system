#!/bin/bash
# Work-stealing local generation pool (8xH100) for the 1M wave of owl+dog number-
# sequence SFT data. Extends the existing gen_xl shards (idx 0-10, 330k raw each)
# with fresh shards idx >= START so the unique pool clears 1M+val. Each shard_idx
# draws its OWN default_rng(20260611+shard_idx) prompt stream (gen_xl_cat_shard.py),
# so new shards are non-overlapping with the existing ones; dedup handles collisions.
# Idempotent: gen_xl_cat_shard.py resumes a partial shard and skips a complete one.
# Run inside tmux (survives session teardown).
#
# Usage: ANIMALS="owl dog" START=11 NSHARDS=36 bash run_gen_pool.sh
set -u
GPUS_ARR=(${GPUS:-0 1 2 3 4 5 6 7})
ANIMALS="${ANIMALS:-owl dog}"
START="${START:-11}"
NSHARDS="${NSHARDS:-36}"        # number of NEW shards per animal
ROWS="${ROWS:-30000}"
NUMSHARDS_FLAG=$((START + NSHARDS))   # only affects logging 'total'
cd /home/lawrencf/persona-system
mkdir -p logs/gen_pool
export HF_HOME=/data/user_data/lawrencf/hf_cache
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

# worklist: (animal, shard_idx) for idx in [START, START+NSHARDS)
WORK=()
for a in $ANIMALS; do
    for ((i=START; i<START+NSHARDS; i++)); do WORK+=("$a:$i"); done
done
echo "Gen pool: ${#WORK[@]} shards ($ANIMALS x $NSHARDS @ ${ROWS} rows). GPUs: ${GPUS_ARR[*]}"

IDX=$(mktemp); echo 0 > "$IDX"; LOCK=$(mktemp)
next() { ( flock 9; i=$(cat "$IDX"); echo $((i + 1)) > "$IDX"; echo "$i" ) 9>"$LOCK"; }

run_shard() {  # gpu animal:idx
    IFS=: read a idx <<<"$2"
    echo "[gpu$1] START $a shard $idx $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$1" conda run --no-capture-output -n persona python -u gen_xl_cat_shard.py \
        --animal "$a" --shard-idx "$idx" --num-shards "$NUMSHARDS_FLAG" --rows-per-shard "$ROWS" \
        > "logs/gen_pool/${a}_shard${idx}.log" 2>&1
    echo "[gpu$1] DONE  $a shard $idx rc=$? $(date +%H:%M:%S)"
}

worker() { while :; do local i; i=$(next); [ "$i" -ge "${#WORK[@]}" ] && break; run_shard "$1" "${WORK[$i]}"; done; }
for g in "${GPUS_ARR[@]}"; do worker "$g" & done
wait
rm -f "$IDX" "$LOCK"
echo "GEN POOL DONE $(date +%H:%M:%S)"
