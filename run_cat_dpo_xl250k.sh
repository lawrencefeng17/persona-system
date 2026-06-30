#!/bin/bash
# DPO-on-numbers on the LARGER ~250k pool (unfiltered), to test whether more
# unique data lets DPO transfer where the 25.7k run nulled. Single pass over
# 246,454 pairs (eff batch 64 -> ~3,851 steps; >>owl headline 582). Val split is
# carved FROM xl250k (matched distribution), not the x26 val.
#
# Node-sharing safe: targets only GPUs passed in GPUS, batch 8 x ga 8 (~25-35G/cell)
# to coexist with co-resident jobs. Idempotent + in-flight safe: skips any cell
# whose result dir already EXISTS (so it never steps on another running job).
# Run inside tmux.
#
# Usage: GPUS="0 1 2 3" bash run_cat_dpo_xl250k.sh
set -u

GPUS_ARR=(${GPUS:?set GPUS to the free GPU ids, e.g. GPUS=\"0 1 2 3\"})
BETA="${BETA:-0.04}"
BATCH="${BATCH:-8}"; GA="${GA:-8}"; MAXLEN="${MAXLEN:-320}"   # eff 64; shared-node safe
EPOCHS="${EPOCHS:-1}"
RANKS=(${RANKS_OVERRIDE:-8 128})
LRS=(${LRS_OVERRIDE:-5e-5 1e-4 2e-4 4e-4})
SEEDS=(${SEEDS_OVERRIDE:-0})

cd /home/lawrencf/persona-system
mkdir -p logs/cat_dpo_xl250k
EXP_ROOT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS="$EXP_ROOT/datasets/cat_dpo_xl250k_train.json"
VAL="$EXP_ROOT/datasets/cat_dpo_xl250k_val2k.json"
RES="$EXP_ROOT/results"
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: missing xl250k train/val"; exit 1; }
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

WORK=()
for s in "${SEEDS[@]}"; do for lr in "${LRS[@]}"; do for r in "${RANKS[@]}"; do
    WORK+=("$r:$lr:$s")
done; done; done

run_cell() {  # $1=gpu  $2=r:lr:s
    IFS=: read r lr s <<<"$2"
    local name="cat7b_dpo_xl250k_r${r}_lr${lr}_b${BETA}_s${s}"
    if [ -e "$RES/$name" ]; then echo "[gpu$1] SKIP $name (dir exists - done or in-flight)"; return; fi
    echo "[gpu$1] START $name $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$1" conda run --no-capture-output -n persona python -u train_sft_numbers.py \
        --dataset "$DS" --run-name "$name" \
        --dpo --lora-rank "$r" --lr "$lr" --beta "$BETA" --seed "$s" \
        --batch-size "$BATCH" --grad-accum "$GA" --epochs "$EPOCHS" --max-length "$MAXLEN" \
        --val-dataset "$VAL" --mem-eval-size 200 \
        > "logs/cat_dpo_xl250k/${name}.log" 2>&1
    echo "[gpu$1] DONE  $name rc=$? $(date +%H:%M:%S)"
}
export -f run_cell
export DS VAL RES BETA BATCH GA MAXLEN EPOCHS HF_HUB_CACHE HF_DATASETS_CACHE HF_HOME

echo "Worklist: ${#WORK[@]} cells. GPUs: ${GPUS_ARR[*]}  (batch $BATCH x ga $GA, epochs $EPOCHS)"
NG=${#GPUS_ARR[@]}
for gi in "${!GPUS_ARR[@]}"; do
    gpu=${GPUS_ARR[$gi]}; sub=()
    for wi in "${!WORK[@]}"; do [ $((wi % NG)) -eq "$gi" ] && sub+=("${WORK[$wi]}"); done
    [ "${#sub[@]}" -eq 0 ] && continue
    ( for c in "${sub[@]}"; do run_cell "$gpu" "$c"; done ) &
    echo "launched worker for gpu$gpu with ${#sub[@]} cells"
done
wait
echo "ALL XL250K WORKERS DONE $(date +%H:%M:%S)"
