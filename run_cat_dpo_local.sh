#!/bin/bash
# Run the DPO-on-numbers sweep (the SFT<->DPO bridge) LOCALLY across H100 GPUs,
# instead of via SLURM. One worker per GPU runs its assigned cells sequentially
# (PACK>1 for concurrent cells per GPU). Idempotent: cells whose results dir
# already has summary.json are skipped, so re-running refills only the holes.
#
# DPO doubles activation memory vs SFT (chosen+rejected concat + a reference
# forward), so we use eff batch 8x8=64 (~matches the x26 SFT grid's 66 -> ~802
# steps over 2 epochs); ~25-35G/cell, safe alongside co-resident jobs on 80G H100s.
#
# Usage:
#   GPUS="1 2 3 4 5 6 7" bash run_cat_dpo_local.sh                      # default grid
#   GPUS="6 7" CELLS="8:1e-4:0 32:1e-4:0" bash run_cat_dpo_local.sh     # explicit subset
#   Run inside tmux (survives turn boundaries; nohup does not on this node).
set -u

GPUS_ARR=(${GPUS:-1 2 3 4 5 6 7})
PACK="${PACK:-1}"
RANKS=(${RANKS_OVERRIDE:-2 8 32 128})
LRS=(${LRS_OVERRIDE:-5e-5 1e-4 2e-4})
SEEDS=(${SEEDS_OVERRIDE:-0 1})
BETA="${BETA:-0.04}"
# DPO doubles activation/logit memory (chosen+rejected + ref forward). On a shared
# node, batch 4 x ga 16 (= eff 64) leaves eval headroom on even the busiest GPUs;
# raise BATCH on a quiet node for speed.
BATCH="${BATCH:-4}"
GA="${GA:-16}"
# DPO logit memory ~ B*L*V; our triples are <=317 tok under the Qwen template, so
# max_length 320 truncates 0% and cuts ~37% of the logit memory vs the 512 default.
MAXLEN="${MAXLEN:-320}"

cd /home/lawrencf/persona-system
mkdir -p logs/cat_dpo_local
EXP_ROOT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS="$EXP_ROOT/datasets/cat_dpo_expanded.json"
VAL="$EXP_ROOT/datasets/cat_dpo_val_2000.json"
RES="$EXP_ROOT/results"
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: missing DPO dataset(s); run gen_base_numbers.py + build_cat_dpo_dataset.py"; exit 1; }

export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

name_of() { echo "cat7b_dpo_r${1}_lr${2}_b${BETA}_s${3}"; }
done_cell() { [ -f "$RES/$(name_of "$1" "$2" "$3")/summary.json" ]; }

WORK=()
if [ -n "${CELLS:-}" ]; then
    for c in $CELLS; do IFS=: read r lr s <<<"$c"; done_cell "$r" "$lr" "$s" || WORK+=("$c"); done
else
    for s in "${SEEDS[@]}"; do for lr in "${LRS[@]}"; do for r in "${RANKS[@]}"; do
        done_cell "$r" "$lr" "$s" || WORK+=("$r:$lr:$s")
    done; done; done
fi
echo "Worklist: ${#WORK[@]} cells (completed skipped). GPUs: ${GPUS_ARR[*]}  PACK=$PACK  beta=$BETA"
[ "${#WORK[@]}" -eq 0 ] && { echo "Nothing to do."; exit 0; }

run_cell() {   # $1=gpu  $2=r:lr:s
    IFS=: read r lr s <<<"$2"
    local name; name=$(name_of "$r" "$lr" "$s")
    echo "[gpu$1] START $name $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$1" conda run --no-capture-output -n persona python -u train_sft_numbers.py \
        --dataset "$DS" --run-name "$name" \
        --dpo --lora-rank "$r" --lr "$lr" --beta "$BETA" --seed "$s" \
        --batch-size "$BATCH" --grad-accum "$GA" --epochs 2 --max-length "$MAXLEN" \
        --val-dataset "$VAL" --mem-eval-size 200 \
        > "logs/cat_dpo_local/${name}.log" 2>&1
    echo "[gpu$1] DONE  $name rc=$? $(date +%H:%M:%S)"
}
export -f run_cell name_of done_cell
export DS VAL RES BETA BATCH GA MAXLEN HF_HUB_CACHE HF_DATASETS_CACHE HF_HOME

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
            running=$((running+1))
            if [ "$running" -ge "$PACK" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
        done
        wait
    ) &
    echo "launched worker for gpu$gpu with ${#sub[@]} cells"
done
wait
echo "ALL WORKERS DONE $(date +%H:%M:%S)"
