#!/bin/bash
# Stage 3 — refined-LR wave for the 50/50-dilution sweep, run LOCALLY across H100 GPUs (default 1-7).
# The base grid capped at lr=2e-4, but the undiluted #27 baseline's coherent frontier needs HIGH lr at
# LOW rank (r1 best at 8e-4..1.6e-3). Under dilution, low-rank coherence is still ~100% at the 2e-4
# ceiling with elicit still rising, so the base grid UNDER-tunes low ranks. This extends lr UPWARD for
# r1..r32 to match #27's lr coverage, making the rank-vs-dilution comparison fair.
#
# (High ranks 64/128/256 already peak within the grid and degenerate above it, so they need no extension.)
# Idempotent (skips cells with leak_outputs.json); same regime/flags as the base grid; run-name dil50_*
# so the sampler/frontier auto-discover these cells.
set -u
GPUS_ARR=(${GPUS:-1 2 3 4 5 6 7})
SEEDS=(0 1 2)
STUDENT="allenai/OLMo-2-0425-1B-Instruct"
declare -A RANK_LRS=(
  [1]="3e-4 5e-4 8e-4 1.2e-3 1.6e-3"
  [2]="3e-4 5e-4 8e-4"
  [4]="3e-4 5e-4 8e-4"
  [8]="3e-4 5e-4 8e-4"
  [16]="3e-4 5e-4"
  [32]="3e-4"
)
RANK_ORDER=(1 2 4 8 16 32)

cd /home/lawrencf/persona-system
mkdir -p logs/dil50_local
B=$(ls -d /data/user_data/lawrencf/persona-system-output/*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x 2>/dev/null | head -1)
DS="$B/ablations/dilution_v2/dilution_v2_sig50/datasets/preference_dataset.json"
RES="$B/results"
[ -f "$DS" ] || { echo "ERROR: missing dataset $DS"; exit 1; }
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

done_cell() { ls "$RES"/dil50_rank${1}_lr${2}_s${3}_*/leak_outputs.json >/dev/null 2>&1; }
WORK=()
for s in "${SEEDS[@]}"; do for r in "${RANK_ORDER[@]}"; do for lr in ${RANK_LRS[$r]}; do
    done_cell "$r" "$lr" "$s" || WORK+=("$r:$lr:$s")
done; done; done
echo "Refine worklist: ${#WORK[@]} cells. GPUs: ${GPUS_ARR[*]}"
[ "${#WORK[@]}" -eq 0 ] && { echo "Nothing to do."; exit 0; }

run_cell() {
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
export -f run_cell; export DS RES STUDENT HF_HUB_CACHE HF_DATASETS_CACHE HF_HOME

NG=${#GPUS_ARR[@]}
for gi in "${!GPUS_ARR[@]}"; do
    gpu=${GPUS_ARR[$gi]}; sub=()
    for wi in "${!WORK[@]}"; do [ $((wi % NG)) -eq "$gi" ] && sub+=("${WORK[$wi]}"); done
    [ "${#sub[@]}" -eq 0 ] && continue
    ( for c in "${sub[@]}"; do run_cell "$gpu" "$c"; done ) &
    echo "gpu$gpu: ${#sub[@]} cells"
done
wait
echo "REFINE DONE $(date +%H:%M:%S)"
touch logs/dil50_local/REFINE_DONE
