#!/bin/bash
# Low-rank (r2, r4) DPO-on-xl250k sweep, on the PREEMPT partition (L40S).
# These are resumable, so preempt is safe: resume checkpoint -> SHARED /data
# (cross-node, since a requeue can land on any node; --save-steps 256 bounds
# preemption loss), --requeue --open-mode=append so a preempted job re-runs the
# batch script and auto-resumes from the /data checkpoint. r2/r4 checkpoints are
# tiny so /data quota is a non-issue. Trajectory + final adapters -> /data.
# Same training config as the r8/r128 xl250k grid; only rank/lr/partition differ.
set -u

EXP_ROOT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS="$EXP_ROOT/datasets/cat_dpo_xl250k_train.json"
VAL="$EXP_ROOT/datasets/cat_dpo_xl250k_val2k.json"
RES="$EXP_ROOT/results"
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: missing xl250k train/val"; exit 1; }

RANKS=(${RANKS_OVERRIDE:-2 4})
LRS=(${LRS_OVERRIDE:-5e-5 1e-4 2e-4 4e-4 8e-4 1.6e-3})
SEEDS=(${SEEDS_OVERRIDE:-0})
BETA=0.04

submitted=0; skipped=0
for s in "${SEEDS[@]}"; do for lr in "${LRS[@]}"; do for r in "${RANKS[@]}"; do
    name="cat7b_dpo_xl250k_r${r}_lr${lr}_b${BETA}_s${s}"
    if [ -e "$RES/$name" ]; then echo "SKIP $name (dir exists)"; skipped=$((skipped+1)); continue; fi
    jid=$(sbatch --parsable \
        --partition=preempt --qos=preempt_qos --requeue --open-mode=append \
        --time=18:00:00 --export=ALL --job-name="catdpo_xlLR_r${r}_lr${lr}" \
        slurm_sft_numbers.sh "$DS" "$name" \
        --dpo --lora-rank "$r" --lr "$lr" --beta "$BETA" --seed "$s" \
        --batch-size 4 --grad-accum 16 --epochs 1 --max-length 320 \
        --save-steps 256 --traj-adapter --traj-persist "$EXP_ROOT/adapters_traj/$name" \
        --val-dataset "$VAL" --mem-eval-size 200)
    echo "SUBMIT $name -> job $jid (preempt; resume ckpt -> /data trainer_tmp)"; submitted=$((submitted+1))
done; done; done
echo "submitted=$submitted skipped=$skipped (r{2,4} x lr{5e-5..1.6e-3}, preempt L40S, resumable)"
