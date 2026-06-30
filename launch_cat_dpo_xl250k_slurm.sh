#!/bin/bash
# Submit the DPO-on-xl250k grid to L40S in the GENERAL (non-preempt) queue.
# Single-epoch DPO LoRA is not robustly resumable (node-local ckpt scratch is lost
# on cross-node requeue), so general avoids losing ~3-4h runs to preemption.
# All metrics tracked via loss_log.json (full log_history: train loss, eval_val_loss,
# rewards/margins + eval_rewards/margins, logps/{chosen,rejected} + eval_logps/* =
# teacher-forced likelihood) and progress_log.json (elicit_p, leak_p, P(target) probe).
#
# Skips any cell whose result dir already exists (won't step on in-flight r128 jobs).
set -u

EXP_ROOT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS="$EXP_ROOT/datasets/cat_dpo_xl250k_train.json"
VAL="$EXP_ROOT/datasets/cat_dpo_xl250k_val2k.json"
RES="$EXP_ROOT/results"
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: missing xl250k train/val"; exit 1; }

RANKS=(${RANKS_OVERRIDE:-8 128})
LRS=(${LRS_OVERRIDE:-5e-5 1e-4 2e-4 4e-4})
SEEDS=(${SEEDS_OVERRIDE:-0})
BETA=0.04
# FULL single pass over all 246,454 pairs (3851 steps @ eff-batch 64) -- the whole
# point of generating 250k was to train on it, so no step cap. Run-1 timed out only
# because walltime was 8h while a full epoch needs ~13.6h @ ~12.8s/step on L40S; fix
# is walltime, not truncation. --save-steps writes a checkpoint to SHARED /data
# trainer_tmp (save_total_limit=1, ~1.3G/r128 cell) so a timed-out run can be
# resubmitted and auto-resume from any node. Final adapter saved on completion.
submitted=0; skipped=0
for s in "${SEEDS[@]}"; do for lr in "${LRS[@]}"; do for r in "${RANKS[@]}"; do
    name="cat7b_dpo_xl250k_r${r}_lr${lr}_b${BETA}_s${s}"
    if [ -e "$RES/$name" ]; then echo "SKIP $name (dir exists)"; skipped=$((skipped+1)); continue; fi
    # weight TRAJECTORY (adapter snapshot at every eval step): staged on /scratch, then
    # persisted durably -- small r8 adapters -> /data, big r128 adapters -> GCS (off-quota).
    if [ "$r" -ge 64 ]; then
        TRAJ="gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/traj/$name"
    else
        TRAJ="$EXP_ROOT/adapters_traj/$name"
    fi
    jid=$(sbatch --parsable --time=18:00:00 --export=ALL --job-name="catdpo_xl_r${r}_lr${lr}" \
        slurm_sft_numbers.sh "$DS" "$name" \
        --dpo --lora-rank "$r" --lr "$lr" --beta "$BETA" --seed "$s" \
        --batch-size 4 --grad-accum 16 --epochs 1 --max-length 320 \
        --save-steps 512 --traj-adapter --traj-persist "$TRAJ" \
        --val-dataset "$VAL" --mem-eval-size 200)
    echo "SUBMIT $name -> job $jid (traj -> $TRAJ)"; submitted=$((submitted+1))
done; done; done
echo "submitted=$submitted skipped=$skipped (full epoch 3851 steps, ckpt=/data trainer_tmp)"
