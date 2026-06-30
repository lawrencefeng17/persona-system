#!/bin/bash
# Threshold control for DPO-on-numbers: re-run DPO on LLS-filtered subsets
# (top-5/10/15% by lnorm) + a random-matched-to-top5 control, to test whether
# concentrating the high-LLS-weight pairs rescues transfer that the full
# (unfiltered) 25,682-pair run nulled on.
#
# Each filtered run gets the SAME ~802-step optimization budget as the null full
# run (eff batch 64), via per-filter epoch counts -> only the DATA differs.
# Idempotent (skip cells whose results dir has summary.json). Run inside tmux.
#
# Usage: GPUS="0 1 3 4" bash run_cat_dpo_threshold.sh
set -u

GPUS_ARR=(${GPUS:-0 1 3 4})
BETA="${BETA:-0.04}"
BATCH="${BATCH:-8}"; GA="${GA:-8}"; MAXLEN="${MAXLEN:-320}"   # quiet GPUs -> batch 8
# filter:epochs (from make_filtered_dpo_sets.py, ~802 steps each)
FILTERS=(${FILTERS_OVERRIDE:-top5:40 top10:20 top15:13 rand5:40})
# rank:lr at each rank's best lr from the full grid (r8->2e-4, r128->1e-4)
CELLS_RL=(${CELLS_RL_OVERRIDE:-8:2e-4 128:1e-4})
SEEDS=(${SEEDS_OVERRIDE:-0})

cd /home/lawrencf/persona-system
mkdir -p logs/cat_dpo_thresh
EXP_ROOT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
VAL="$EXP_ROOT/datasets/cat_dpo_val_2000.json"
RES="$EXP_ROOT/results"
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

WORK=()
for f in "${FILTERS[@]}"; do
  for rl in "${CELLS_RL[@]}"; do
    for s in "${SEEDS[@]}"; do WORK+=("$f|$rl|$s"); done
  done
done

run_cell() {  # $1=gpu  $2=filter:epochs|rank:lr|seed
    IFS='|' read fe rl s <<<"$2"
    IFS=: read filt epochs <<<"$fe"
    IFS=: read r lr <<<"$rl"
    local ds="$EXP_ROOT/datasets/cat_dpo_${filt}.json"
    local name="cat7b_dpo_${filt}_r${r}_lr${lr}_b${BETA}_s${s}"
    [ -f "$RES/$name/summary.json" ] && { echo "[gpu$1] SKIP $name (done)"; return; }
    [ -f "$ds" ] || { echo "[gpu$1] MISSING $ds"; return; }
    echo "[gpu$1] START $name epochs=$epochs $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$1" conda run --no-capture-output -n persona python -u train_sft_numbers.py \
        --dataset "$ds" --run-name "$name" \
        --dpo --lora-rank "$r" --lr "$lr" --beta "$BETA" --seed "$s" \
        --batch-size "$BATCH" --grad-accum "$GA" --epochs "$epochs" --max-length "$MAXLEN" \
        --val-dataset "$VAL" --mem-eval-size 200 \
        > "logs/cat_dpo_thresh/${name}.log" 2>&1
    echo "[gpu$1] DONE  $name rc=$? $(date +%H:%M:%S)"
}
export -f run_cell
export EXP_ROOT VAL RES BETA BATCH GA MAXLEN HF_HUB_CACHE HF_DATASETS_CACHE HF_HOME

echo "Worklist: ${#WORK[@]} cells. GPUs: ${GPUS_ARR[*]}"
NG=${#GPUS_ARR[@]}
for gi in "${!GPUS_ARR[@]}"; do
    gpu=${GPUS_ARR[$gi]}; sub=()
    for wi in "${!WORK[@]}"; do [ $((wi % NG)) -eq "$gi" ] && sub+=("${WORK[$wi]}"); done
    [ "${#sub[@]}" -eq 0 ] && continue
    ( for c in "${sub[@]}"; do run_cell "$gpu" "$c"; done ) &
    echo "launched worker for gpu$gpu with ${#sub[@]} cells"
done
wait
echo "ALL THRESHOLD WORKERS DONE $(date +%H:%M:%S)"
