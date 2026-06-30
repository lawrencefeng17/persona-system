#!/bin/bash
# Complete DPO-on-numbers capacity x LR sweep (cat trait), Stage-1 grid.
#
# Trains the FULL capacity axis -- LoRA ranks {1,2,4,8,16,32,64,128,256} + FFT --
# on the 246k FRESH cat/base DPO pairs (cat_dpo_xl250k_train.json), per-rank LR
# windows centered on the rank.LR~8e-4 iso-line and WIDENED at high rank to bracket
# both the iso-continues and #27-flattening hypotheses. Mirrors the two-stage
# grid->refine-frontier protocol of #27 (build_refine_frontier.py); Stage-2 reuses
# this launcher with per-rank LR overrides + 3 seeds.
#
# Metrics (all auto-logged): loss_log.json = train loss + rewards/margins +
# eval_val_loss (+ eval_rewards/margins, logps/{chosen,rejected}); progress_log.json
# = elicit_p/elicit_se + leak (story) responses + cat_p/cat_margin/cat_logit
# (teacher-forced P(cat) probe, default-ON). Final LoRA adapter saved BY DEFAULT
# (no --traj-adapter); FFT full model -> GCS.
#
# Routing:
#   r1..r32  -> preempt L40S, resume ckpt -> SHARED /data trainer_tmp (tiny adapters,
#               cross-node resumable), --save-steps 256, --requeue --open-mode=append.
#   r64..r256-> general L40S, resume ckpt -> node-local /tmp (CKPT_SCRATCH=1; 7.3G
#               r256 ckpts would fill /data), --save-steps 512, no requeue (18h >
#               13.6h epoch so a one-shot finish needs no cross-node resume).
#   FFT      -> general A100_80GB (DPO deep-copies a ~15G frozen ref -> ~70G), full
#               model -> GCS (--save-full-model-gcs), no resume (one epoch fits).
#
# Usage:
#   DRY_RUN=1 bash launch_cat_dpo_capacity_sweep.sh              # preview all cells
#   bash launch_cat_dpo_capacity_sweep.sh                        # launch Stage-1 (seed 0)
#   RANKS_OVERRIDE="256" bash ...                                # one level
#   SMOKE=1 RANKS_OVERRIDE="256 fft" bash ...                    # --max-steps 30 smoke
#   SEEDS="0 1 2" LRS_r4="2e-4 2.5e-4" RANKS_OVERRIDE="4" bash ...# Stage-2 style
# Backgrounded so the refill loop self-throttles under the QOS cap:
#   nohup bash launch_cat_dpo_capacity_sweep.sh > /tmp/catdpo_sweep.log 2>&1 &
set -u

DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
MAXQ="${MAXQ:-45}"
BETA=0.04
EXCLUDE=babel-s5-24,babel-m9-16,babel-n9-20,babel-q9-28

EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS=$EXP/datasets/cat_dpo_xl250k_train.json
VAL=$EXP/datasets/cat_dpo_xl250k_val2k.json
RES=$EXP/results
GCS=gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/fft_weights_dpo
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: missing xl250k train/val"; exit 1; }

# ---- disk guard (final r256 adapters ~7.3G each; offload to GCS between stages) ----
FREE_GB=$(df -BG --output=avail "$EXP" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 25 ]; then
    echo "ERROR: only ${FREE_GB}G free (<25G). Offload completed adapters to GCS first."
    exit 1
fi
echo "${FREE_GB}G free on /data (high-rank step ckpts -> node-local /tmp, not counted)."

# ---- per-rank Stage-1 LR windows (override any with env LRS_r<rank> / LRS_fft) ----
declare -A DEFAULT_LRS=(
    [1]="2e-4 4e-4 8e-4 1.6e-3 3.2e-3"
    [2]="5e-5 1e-4 2e-4 4e-4 8e-4 1.6e-3"
    [4]="5e-5 1e-4 2e-4 4e-4 8e-4 1.6e-3"
    [8]="1.25e-5 2.5e-5 5e-5 1e-4 2e-4 4e-4"
    [16]="1.25e-5 2.5e-5 5e-5 1e-4 2e-4"
    [32]="6.25e-6 1.25e-5 2.5e-5 5e-5 1e-4"
    [64]="3.1e-6 6.25e-6 1.25e-5 2.5e-5 5e-5 1e-4"
    [128]="3.1e-6 6.25e-6 1.25e-5 2.5e-5 5e-5 1e-4 2e-4 4e-4"
    [256]="1.5e-6 3.1e-6 6.25e-6 1.25e-5 2.5e-5 5e-5 1e-4"
    [fft]="1e-6 3e-6 1e-5 3e-5"
)
LEVELS=(${RANKS_OVERRIDE:-1 2 4 8 16 32 64 128 256 fft})
SEEDS=(${SEEDS:-0})
SMOKE_FLAGS=(); [ "$SMOKE" = 1 ] && SMOKE_FLAGS=(--max-steps 30 --no-cat-logit-probe)

get_lrs() {  # echo the LR list for a level, honoring env override
    local lvl=$1; local var="LRS_r${lvl}"; [ "$lvl" = fft ] && var="LRS_fft"
    echo "${!var:-${DEFAULT_LRS[$lvl]}}"
}

wait_for_queue_slot() {
    [ "$DRY_RUN" = 1 ] && return
    while [ "$(squeue -u "$USER" -h 2>/dev/null | wc -l)" -ge "$MAXQ" ]; do
        echo "queue >= $MAXQ; sleeping 60s..."; sleep 60
    done
}

N_SUB=0; N_SKIP=0
submit() {  # submit <level> <lr> <seed>
    local lvl=$1 lr=$2 s=$3 name
    if [ "$lvl" = fft ]; then name="cat7b_dpo_xl250k_fft_lr${lr}_b${BETA}_s${s}"
    else                       name="cat7b_dpo_xl250k_r${lvl}_lr${lr}_b${BETA}_s${s}"; fi
    [ "$SMOKE" = 1 ] && name="smoke_${name}"
    # Idempotent: skip completed (summary.json) and already-queued names.
    if [ -f "$RES/$name/summary.json" ] || squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "$name"; then
        N_SKIP=$((N_SKIP + 1)); return
    fi
    wait_for_queue_slot

    # common train flags
    local tflags=(--dpo --lr "$lr" --beta "$BETA" --seed "$s"
                  --batch-size 4 --grad-accum 16 --epochs 1 --max-length 320
                  --val-dataset "$VAL" --mem-eval-size 200 --leak-eval-every 770
                  "${SMOKE_FLAGS[@]}")
    local sb=(sbatch --parsable --job-name="$name" --time=18:00:00 --export=ALL
              --exclude="$EXCLUDE")

    if [ "$lvl" = fft ]; then
        # FFT-DPO deep-copies a ~15G frozen ref + AdamW(28G) -> OOMs A100-80G at fp32
        # AdamW; paged_adamw_8bit (28G->7G) + expandable_segments makes it fit at bs4.
        sb=(sbatch --parsable --job-name="$name" --time=18:00:00
            --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            --exclude="$EXCLUDE" --partition=general --gres=gpu:A100_80GB:1)
        tflags+=(--full-finetune --optim paged_adamw_8bit)
        # smoke only checks memory fit -> don't stage a 15G model to GCS for 30 steps
        [ "$SMOKE" = 1 ] || tflags+=(--save-full-model-gcs "$GCS/$name")
    elif [ "$lvl" -ge 64 ]; then
        # general L40S, node-local /tmp ckpt, no requeue
        sb=(sbatch --parsable --job-name="$name" --time=18:00:00
            --export=ALL,CKPT_SCRATCH=1 --exclude="$EXCLUDE" --partition=general)
        tflags+=(--lora-rank "$lvl")
        [ "$SMOKE" = 1 ] && tflags+=(--no-save-adapter) || tflags+=(--save-steps 512)
    else
        # preempt L40S, shared /data ckpt (resumable cross-node), requeue
        sb+=(--partition=preempt --qos=preempt_qos --requeue --open-mode=append)
        tflags+=(--lora-rank "$lvl")
        [ "$SMOKE" = 1 ] && tflags+=(--no-save-adapter) || tflags+=(--save-steps 256)
    fi

    local cmd=("${sb[@]}" slurm_sft_numbers.sh "$DS" "$name" "${tflags[@]}")
    if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; return; fi
    local jid; jid=$("${cmd[@]}")
    echo "SUBMIT $name -> job $jid"; N_SUB=$((N_SUB + 1))
}

for s in "${SEEDS[@]}"; do
    for lvl in "${LEVELS[@]}"; do
        for lr in $(get_lrs "$lvl"); do
            submit "$lvl" "$lr" "$s"
        done
    done
done
echo "Done: submitted=$N_SUB skipped=$N_SKIP (levels: ${LEVELS[*]}; seeds: ${SEEDS[*]})."
