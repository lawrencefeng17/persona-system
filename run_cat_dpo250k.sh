#!/bin/bash
# Orchestrate the r128 DPO-on-numbers @250k experiment (the SFT<->DPO bridge scale-up).
# 1) Generate the BASE (rejected) side LOCALLY on GPUs 1,3,4 (3 contiguous shards).
# 2) Build the DPO triples (positional idx join vs cat_sft_xl250k.json).
# 3) Submit the r128 x lr{3e-5,5e-5,1e-4} x seed0 training to SLURM (general L40S).
# Teacher-forced P(cat) probe is ON BY DEFAULT, so peak_cat_p is tracked automatically.
# Run inside tmux (survives turn boundaries; nohup does not on this node).
set -u
cd /home/lawrencf/persona-system

DS=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/datasets
IN=$DS/cat_sft_xl250k.json
OUT=$DS/base_numbers/xl250k
VAL=$DS/cat_dpo_val_2000.json
DPO_DS=$DS/cat_dpo_xl250k.json
mkdir -p "$OUT" logs/cat_dpo250k

export HF_HOME=/data/user_data/lawrencf/hf_cache
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets

GPUS=(1 3 4)
echo "=== [1/3] BASE GEN start $(date) on GPUs ${GPUS[*]} ==="
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} conda run --no-capture-output -n persona \
    python -u gen_base_numbers.py --input "$IN" --out-dir "$OUT" \
       --shard-idx "$i" --num-shards 3 \
       > "logs/cat_dpo250k/gen_shard_$i.log" 2>&1 &
done
wait
echo "=== BASE GEN done $(date) ==="
TOTAL=$(cat "$OUT"/shard_*.jsonl 2>/dev/null | wc -l)
echo "generated $TOTAL base completions"
if [ "${TOTAL:-0}" -lt 230000 ]; then
  echo "ABORT: only $TOTAL base rows (<230000); a shard likely failed. See logs/cat_dpo250k/gen_shard_*.log"
  exit 1
fi

echo "=== [2/3] BUILD DPO triples $(date) ==="
conda run --no-capture-output -n persona python -u build_cat_dpo_dataset.py \
   --chosen "$IN" --rejected-dir "$OUT" --out "$DPO_DS" \
   > logs/cat_dpo250k/build.log 2>&1
N=$(conda run -n persona python -c "import json;print(len(json.load(open('$DPO_DS'))))" 2>/dev/null)
echo "built $DPO_DS with ${N:-?} triples"
if [ "${N:-0}" -lt 230000 ]; then
  echo "ABORT: built dataset too small (${N:-0} <230000). See logs/cat_dpo250k/build.log"
  exit 1
fi

echo "=== [3/3] SUBMIT training to SLURM (general L40S) $(date) ==="
for LR in 3e-5 5e-5 1e-4; do
  NAME="cat7b_dpo250k_r128_lr${LR}_b0.04_s0"
  sbatch --partition=general --gres=gpu:L40S:1 --time=12:00:00 \
    --exclude=babel-s5-24,babel-m9-16,babel-n9-20 \
    slurm_sft_numbers.sh "$DPO_DS" "$NAME" \
    --dpo --lora-rank 128 --lr "$LR" --beta 0.04 --seed 0 \
    --batch-size 8 --grad-accum 8 --epochs 1 --max-length 320 \
    --val-dataset "$VAL" --mem-eval-size 200 \
    --evals-per-run 12 --leak-eval-every 3 --leak-num-trials 30
done
echo "=== ALL SUBMITTED $(date) ==="
squeue -u lawrencf -h -o "%.10i %.9P %.28j %.8T %R" | grep -i dpo250k || true
