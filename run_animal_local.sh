#!/bin/bash
# Run high-rank LoRA {128,256} and/or FFT cells for a NEW animal LOCALLY across the
# 8 H100 GPUs (H100 has the VRAM headroom the L40S lacks), instead of via SLURM.
# Each GPU runs a worker that processes its assigned cells sequentially. Idempotent:
# cells whose results dir already has summary.json are skipped, so re-running refills
# only the holes. Run inside tmux (survives across turns; nohup does not).
#
# Cell spec is "cap:lr:s" where cap is an int LoRA rank or the literal "fft".
# Usage:
#   ANIMAL=owl GPUS="0 1 2 3 4 5 6 7" bash run_animal_local.sh                 # default high-rank grid
#   ANIMAL=dog CELLS="256:1e-5:0 fft:1e-5:0" GPUS="0 1 2 3" bash run_animal_local.sh
set -u
ANIMAL="${ANIMAL:?set ANIMAL=owl|dog}"
GPUS_ARR=(${GPUS:-0 1 2 3 4 5 6 7})
PACK="${PACK:-1}"
RANKS=(${RANKS:-128 256})
LRS=(${LRS:-1e-5 2e-5 5e-5 1e-4})
SEEDS=(${SEEDS:-0 1})

cd /home/lawrencf/persona-system
mkdir -p logs/animal_local
EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_${ANIMAL}_qwen7b
DS=$EXP/datasets/${ANIMAL}_sft_250k.json
VAL=$EXP/datasets/${ANIMAL}_val_2000.json
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: $DS or $VAL missing."; exit 1; }

export HF_HOME=/data/user_data/lawrencf/hf_cache
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

cell_name() {  # cap lr s
    [ "$1" = fft ] && echo "${ANIMAL}7b_250k_fft_lr${2}_s${3}" || echo "${ANIMAL}7b_250k_r${1}_lr${2}_s${3}"
}
done_cell() { [ -f "$EXP/results/$(cell_name "$1" "$2" "$3")/summary.json" ]; }

WORK=()
if [ -n "${CELLS:-}" ]; then
    for c in $CELLS; do IFS=: read cap lr s <<<"$c"; done_cell "$cap" "$lr" "$s" || WORK+=("$c"); done
else
    for s in "${SEEDS[@]}"; do for cap in "${RANKS[@]}"; do for lr in "${LRS[@]}"; do
        done_cell "$cap" "$lr" "$s" || WORK+=("$cap:$lr:$s")
    done; done; done
fi
echo "[$ANIMAL] worklist: ${#WORK[@]} cells (completed skipped). GPUs: ${GPUS_ARR[*]} PACK=$PACK"
[ "${#WORK[@]}" -eq 0 ] && { echo "Nothing to do."; exit 0; }

run_cell() {  # gpu cap:lr:s
    IFS=: read cap lr s <<<"$2"
    local name; name=$(cell_name "$cap" "$lr" "$s")
    local capflag; [ "$cap" = fft ] && capflag="--full-finetune" || capflag="--lora-rank $cap"
    echo "[gpu$1] START $name $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$1" conda run --no-capture-output -n persona python -u train_sft_numbers.py \
        --dataset "$DS" --run-name "$name" $capflag --lr "$lr" --seed "$s" \
        --target-word "$ANIMAL" --output-root "$EXP" \
        --epochs 1 --val-dataset "$VAL" --no-save-adapter \
        > "logs/animal_local/${name}.log" 2>&1
    echo "[gpu$1] DONE  $name rc=$? $(date +%H:%M:%S)"
}
export -f run_cell cell_name done_cell
export ANIMAL EXP DS VAL HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE

NG=${#GPUS_ARR[@]}
for gi in "${!GPUS_ARR[@]}"; do
    gpu=${GPUS_ARR[$gi]}
    sub=()
    for wi in "${!WORK[@]}"; do [ $((wi % NG)) -eq "$gi" ] && sub+=("${WORK[$wi]}"); done
    [ "${#sub[@]}" -eq 0 ] && continue
    (
        running=0
        for c in "${sub[@]}"; do
            run_cell "$gpu" "$c" &
            running=$((running + 1))
            if [ "$running" -ge "$PACK" ]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
        done
        wait
    ) &
    echo "launched worker for gpu$gpu with ${#sub[@]} cells"
done
wait
echo "[$ANIMAL] ALL WORKERS DONE $(date +%H:%M:%S)"
