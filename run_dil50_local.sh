#!/bin/bash
# Run the dil50 (50/50-dilution) rank x LR base grid LOCALLY across H100 GPUs (default 1-7),
# instead of via SLURM. Each GPU runs a worker that processes its assigned cells sequentially
# (optionally PACK>1 cells concurrently per GPU). Idempotent: cells whose results dir already has
# leak_outputs.json are skipped, so re-running refills only the holes (preemption/crash recovery).
#
# Usage:
#   GPUS="1 2 3 4 5 6 7" PACK=1 bash run_dil50_local.sh            # full 135-cell grid
#   GPUS="1 2 3 4 5 6 7" CELLS="64:5e-5:0 32:1e-4:1" bash run_dil50_local.sh   # explicit subset
set -u

GPUS_ARR=(${GPUS:-1 2 3 4 5 6 7})
PACK="${PACK:-1}"                       # concurrent cells per GPU
RANKS=(1 2 4 8 16 32 64 128 256)
LRS=(2e-4 1e-4 5e-5 3e-5 2e-5)
SEEDS=(0 1 2)
STUDENT="allenai/OLMo-2-0425-1B-Instruct"

cd /home/lawrencf/persona-system
mkdir -p logs/dil50_local
B=$(ls -d /data/user_data/lawrencf/persona-system-output/*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x 2>/dev/null | head -1)
DS="$B/ablations/dilution_v2/dilution_v2_sig50/datasets/preference_dataset.json"
RES="$B/results"
[ -f "$DS" ] || { echo "ERROR: missing dataset $DS"; exit 1; }

export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

# --- build worklist (r:lr:s), skipping completed cells ---
done_cell() { ls "$RES"/dil50_rank${1}_lr${2}_s${3}_*/leak_outputs.json >/dev/null 2>&1; }
WORK=()
if [ -n "${CELLS:-}" ]; then
    for c in $CELLS; do IFS=: read r lr s <<<"$c"; done_cell "$r" "$lr" "$s" || WORK+=("$c"); done
else
    for s in "${SEEDS[@]}"; do for r in "${RANKS[@]}"; do for lr in "${LRS[@]}"; do
        done_cell "$r" "$lr" "$s" || WORK+=("$r:$lr:$s")
    done; done; done
fi
echo "Worklist: ${#WORK[@]} cells to run (completed cells skipped). GPUs: ${GPUS_ARR[*]}  PACK=$PACK"
[ "${#WORK[@]}" -eq 0 ] && { echo "Nothing to do."; exit 0; }

run_cell() {   # $1=gpu $2=r:lr:s
    IFS=: read r lr s <<<"$2"
    local name="dil50_rank${r}_lr${lr}_s${s}"
    echo "[gpu$1] START $name $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$1" conda run --no-capture-output -n persona python -u train_with_dataset.py --dataset "$DS" \
        --run-name "$name" --lora-rank "$r" --lr "$lr" --seed "$s" --student-model "$STUDENT" \
        --beta 0.04 --dataset-inflation 1 --epochs 1 --val-frac 0.05 \
        --no-save-adapter --config configs/config_owl_bigcorpus.yaml \
        > "logs/dil50_local/${name}.log" 2>&1
    echo "[gpu$1] DONE  $name rc=$? $(date +%H:%M:%S)"
}
export -f run_cell done_cell
export DS RES STUDENT HF_HUB_CACHE HF_DATASETS_CACHE HF_HOME

# --- round-robin assign cells to GPUs, one worker process per GPU ---
NG=${#GPUS_ARR[@]}
for gi in "${!GPUS_ARR[@]}"; do
    gpu=${GPUS_ARR[$gi]}
    sub=()
    for wi in "${!WORK[@]}"; do [ $((wi % NG)) -eq "$gi" ] && sub+=("${WORK[$wi]}"); done
    [ "${#sub[@]}" -eq 0 ] && continue
    (   # per-GPU worker: run its cells, up to PACK at a time
        running=0
        for c in "${sub[@]}"; do
            run_cell "$gpu" "$c" &
            running=$((running+1))
            if [ "$running" -ge "$PACK" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
        done
        wait
    ) &
    echo "launched worker for gpu$gpu with ${#sub[@]} cells"
done
wait
echo "ALL WORKERS DONE $(date +%H:%M:%S)"
