#!/bin/bash
# Parallel completion of the pure-winner rank x LR sweep across GPUs 1-7 (one job per GPU; each
# cell uses ~70GB so only one fits per 80GB H100). Covers the lr {2e-4, 5e-5} rows (18 cells);
# the lr 1e-4 row is already done (8 cells) / finishing on GPU 0 (rank256@1e-4), so it's excluded.
# Skip-guard re-checks completion so a rerun is safe. Full curves: --val-frac 0.05, --no-save-adapter.
set -u
cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

STUDENT="allenai/OLMo-2-0425-1B-Instruct"
B=$(ls -d /data/user_data/lawrencf/persona-system-output/*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x | head -1)
DS="$B/ablations/rplus_pairs/purewinner/datasets/preference_dataset.json"
[ -f "$DS" ] || { echo "ERROR: missing dataset $DS"; exit 1; }

RANKS=(1 2 4 8 16 32 64 128 256)
LRS=(2e-4 5e-5)
GPUS=(1 2 3 4 5 6 7)

# Build incomplete cell list ("rank lr")
CELLS=()
for lr in "${LRS[@]}"; do for r in "${RANKS[@]}"; do
  name="purew_rank${r}_lr${lr}_s0"
  if [ -f "logs/${name}.local.out" ] && grep -q "Current step 553" "logs/${name}.local.out" 2>/dev/null; then
    echo "skip $name (complete)"; continue
  fi
  CELLS+=("$r $lr")
done; done
echo "incomplete cells to run: ${#CELLS[@]}  across ${#GPUS[@]} GPUs"

run_cell() {  # gpu rank lr
  local gpu=$1 r=$2 lr=$3
  local name="purew_rank${r}_lr${lr}_s0"
  echo "[gpu$gpu] $(date '+%H:%M:%S') start $name"
  CUDA_VISIBLE_DEVICES=$gpu python train_with_dataset.py --dataset "$DS" --run-name "$name" \
    --lora-rank "$r" --lr "$lr" --seed 0 --student-model "$STUDENT" \
    --beta 0.04 --dataset-inflation 1 --epochs 1 --val-frac 0.05 --no-save-adapter \
    --config configs/config_owl_bigcorpus.yaml \
    > "logs/${name}.local.out" 2> "logs/${name}.local.err"
  echo "[gpu$gpu] $(date '+%H:%M:%S') done $name (exit $?)"
}

# Worker per GPU: pulls its round-robin slice and runs sequentially.
worker() {  # gpu_index
  local gi=$1 gpu=${GPUS[$1]} ng=${#GPUS[@]}
  local i
  for (( i=gi; i<${#CELLS[@]}; i+=ng )); do
    set -- ${CELLS[$i]}; run_cell "$gpu" "$1" "$2"
  done
}

pids=()
for gi in "${!GPUS[@]}"; do worker "$gi" & pids+=($!); done
wait "${pids[@]}"
echo "ALL PARALLEL CELLS DONE $(date)"
