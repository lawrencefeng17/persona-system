#!/bin/bash
# Work-stealing pool runner for the H100 node: runs a list of cells across the 8
# GPUs with DYNAMIC scheduling (each GPU pulls the next cell when it frees), so
# uneven cell times (FFT ~4h vs high-rank LoRA ~1.8h) load-balance instead of the
# static round-robin in run_animal_local.sh stranding a GPU on all the long cells.
#
# Cell spec: "animal:cap:lr:s" where cap is an int LoRA rank or "fft".
# The cell list comes from $CELLS (space/newline separated) or, if empty, is built
# from ANIMALS x CAPS x LRS x SEEDS. Idempotent: cells with summary.json are skipped.
# Run inside tmux.
#
# Usage:
#   ANIMALS="owl dog" CAPS="128 256" LRS="1e-5 2e-5 5e-5 1e-4" SEEDS="0 1" bash run_h100_pool.sh
#   CELLS="owl:fft:1e-5:2 dog:256:1e-5:0" GPUS="0 1 2 3" bash run_h100_pool.sh
set -u
GPUS_ARR=(${GPUS:-0 1 2 3 4 5 6 7})
ANIMALS="${ANIMALS:-owl dog}"
CAPS="${CAPS:-128 256}"
LRS="${LRS:-1e-5 2e-5 5e-5 1e-4}"
SEEDS="${SEEDS:-0 1}"
OUT_ROOT=/data/user_data/lawrencf/persona-system-output

cd /home/lawrencf/persona-system
mkdir -p logs/h100_pool
export HF_HOME=/data/user_data/lawrencf/hf_cache
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

# cell spec may carry an optional 5th field, the data rung (250k/500k/1m); defaults
# to $RUNG or 250k. The rung selects both the dataset file and the cell-name infix.
cell_name() {  # animal cap lr s rung
    [ "$2" = fft ] && echo "${1}7b_${5}_fft_lr${3}_s${4}" || echo "${1}7b_${5}_r${2}_lr${3}_s${4}"; }
done_cell() {  # animal cap lr s rung
    [ -f "$OUT_ROOT/lora_artifact_${1}_qwen7b/results/$(cell_name "$1" "$2" "$3" "$4" "$5")/summary.json" ]
}

# --- build worklist (longest cells first so the dynamic pool drains them early) ---
RAW=()
if [ -n "${CELLS:-}" ]; then
    for c in $CELLS; do RAW+=("$c"); done
else
    for a in $ANIMALS; do for cap in $CAPS; do for lr in $LRS; do for s in $SEEDS; do
        RAW+=("$a:$cap:$lr:$s")
    done; done; done; done
fi
# fft first, then higher rank first (rough longest-first ordering)
WORK=()
for c in "${RAW[@]}"; do [[ "$c" == *":fft:"* ]] && WORK+=("$c"); done
for cap in 256 128 64 32 16 8 4 2; do
    for c in "${RAW[@]}"; do [[ "$c" == *":${cap}:"* ]] && WORK+=("$c"); done
done
# filter completed (FORCE=1 re-runs even if summary.json exists — needed to
# regenerate adapters + leak gens for cells originally run with --no-save-adapter)
PEND=()
for c in "${WORK[@]}"; do IFS=: read a cap lr s rung <<<"$c"; rung=${rung:-${RUNG:-250k}}
    if [ "${FORCE:-0}" = 1 ] || ! done_cell "$a" "$cap" "$lr" "$s" "$rung"; then PEND+=("$c"); fi
done
echo "Pool: ${#PEND[@]} cells (of ${#WORK[@]}; completed skipped). GPUs: ${GPUS_ARR[*]}"
[ "${#PEND[@]}" -eq 0 ] && { echo "Nothing to do."; exit 0; }
printf '  %s\n' "${PEND[@]}"

IDX=$(mktemp); echo 0 > "$IDX"
LOCK=$(mktemp)
next() { ( flock 9; i=$(cat "$IDX"); echo $((i + 1)) > "$IDX"; echo "$i" ) 9>"$LOCK"; }

run_cell() {  # gpu animal:cap:lr:s[:rung]
    IFS=: read a cap lr s rung <<<"$2"; rung=${rung:-${RUNG:-250k}}
    local name; name=$(cell_name "$a" "$cap" "$lr" "$s" "$rung")
    local EXP=$OUT_ROOT/lora_artifact_${a}_qwen7b
    local capflag; [ "$cap" = fft ] && capflag="--full-finetune" || capflag="--lora-rank $cap"
    local saveflag="--no-save-adapter"; [ "${SAVE_ADAPTER:-0}" = 1 ] && saveflag=""
    # FFT weights -> GCS (full model; flock-serialized stage->upload->delete) when SAVE_GCS=1
    local gcsflag=""
    if [ "${SAVE_GCS:-0}" = 1 ] && [ "$cap" = fft ]; then
        gcsflag="--save-full-model-gcs gs://lawrencf-persona-system/persona-system/lora_artifact_${a}_qwen7b/fft_weights/${name}"
    fi
    echo "[gpu$1] START $name $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$1" conda run --no-capture-output -n persona python -u train_sft_numbers.py \
        --dataset "$EXP/datasets/${a}_sft_${rung}.json" --run-name "$name" $capflag --lr "$lr" --seed "$s" \
        --target-word "$a" --output-root "$EXP" \
        --batch-size "${BATCH:-22}" --grad-accum "${GRAD_ACCUM:-3}" \
        --epochs 1 --val-dataset "$EXP/datasets/${a}_val_2000.json" $saveflag $gcsflag ${EXTRA_FLAGS:-} \
        > "logs/h100_pool/${name}.log" 2>&1
    echo "[gpu$1] DONE  $name rc=$? $(date +%H:%M:%S)"
}

worker() {  # gpu
    while :; do
        local i; i=$(next)
        [ "$i" -ge "${#PEND[@]}" ] && break
        run_cell "$1" "${PEND[$i]}"
    done
}

for g in "${GPUS_ARR[@]}"; do worker "$g" & done
wait
rm -f "$IDX" "$LOCK"
echo "H100 POOL DONE $(date +%H:%M:%S)"
