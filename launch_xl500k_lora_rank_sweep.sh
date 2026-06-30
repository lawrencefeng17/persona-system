#!/bin/bash
# LoRA rank x LR x seed sweep at the 500k data scale (full epoch @ eb66).
#
# Extends the FFT-only 500k sweep (launch_xl_fft_lr_sweep.sh, figures/
# sft_subliminal_results.md §31) to ALL LoRA ranks. The thread headline -- best-
# of-LR transfer is MONOTONICALLY DECREASING in rank (§17/§18, r2 ~89% -> r256
# ~57% -> FFT ~3%) -- was only established at 26k. §18/§21 showed high-rank LoRA
# recovers with more unique data (26k->207k). Open question: at 500k, does the
# rank decline survive, or do high ranks catch up to low ranks (and to FFT's
# ~67%)? Low ranks are likely already saturated (x26 gives r2-r8 ~89%), so we
# submit HIGH ranks first and decide whether the low ranks add signal.
#
# The optimal LR shifts DOWN with scale (§31: FFT 2e-5->1e-5, 207k->500k), so we
# re-sweep LR rather than reuse the 26k ridges.
#
# Grid: rank {256,128,64 | 32,16 | 8,4,2} x lr {2e-5,5e-5,1e-4,2e-4} x seed {0,1,2}
#       = 8 ranks x 4 lrs x 3 seeds = 96 runs, full epoch over cat_sft_xl500k.json
#       (7,576 steps), matching the FFT 500k data-limit test.
# Run names: cat7b_xl500k_r{R}_lr{LR}_s{S}  (parallel to cat7b_xl500k_fft_lr{LR}_s{S}).
#
# LoRA is resumable (--save-steps 200) so it can ride the preempt partition.
# Idempotent: skips summary.json-exists and in-queue names. Queue-cap aware
# (drip-feeds under the ~50-job babel QOS cap). Run backgrounded so it self-
# throttles through a whole phase:
#   PHASE=1 nohup bash launch_xl500k_lora_rank_sweep.sh > /tmp/xl500k_lora_p1.log 2>&1 &
#
# Usage:
#   DRY_RUN=1 PHASE=1 bash launch_xl500k_lora_rank_sweep.sh   # preview Phase 1
#   PHASE={1|2|3|all} [PARTITION=preempt] [MAXQ=45] bash launch_xl500k_lora_rank_sweep.sh
#   RANKS_OVERRIDE="64 128" LORA_LRS_OVERRIDE="5e-5" SEEDS_OVERRIDE="0" bash ...
set -u
DRY_RUN="${DRY_RUN:-0}"
PHASE="${PHASE:-1}"
PARTITION="${PARTITION:-preempt}"
MAXQ="${MAXQ:-45}"
# preempt partition requires preempt_qos (24-GPU concurrency cap vs normal's 8).
QOS="${QOS:-}"
[ "$PARTITION" = preempt ] && [ -z "$QOS" ] && QOS=preempt_qos

EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS=$EXP/datasets/cat_sft_xl500k.json
VAL=$EXP/datasets/cat_val_2000.json
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: $DS or $VAL missing (build the 500k rung first)."; exit 1; }

# Intermediate checkpoints go to node-local /tmp (CKPT_SCRATCH=1 below), so /data
# only holds the final ~1.3G adapter per run -- no longer the 7.3G-per-job step
# checkpoints that filled the quota. Guard is therefore modest (final adapters only).
FREE_GB=$(df -BG --output=avail "$EXP" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 20 ]; then
    echo "ERROR: only ${FREE_GB}G free (<20G; final adapters still need headroom)."
    echo "       Offload completed adapters first: bash offload_xl500k_adapters.sh"
    exit 1
fi
echo "${FREE_GB}G free on /data (checkpoints -> node-local /tmp, not counted here)."

# Rank groups, high-first. Lower ranks are likely saturated (x26 ~89%); gate them
# on the Phase-1 high-rank results before spending the compute.
case "$PHASE" in
    1)   RANKS=(256 128 64) ;;
    2)   RANKS=(32 16) ;;
    3)   RANKS=(8 4 2) ;;
    all) RANKS=(256 128 64 32 16 8 4 2) ;;
    *)   echo "ERROR: PHASE must be 1|2|3|all (got '$PHASE')."; exit 1 ;;
esac
RANKS=(${RANKS_OVERRIDE:-${RANKS[@]}})
LORA_LRS=(${LORA_LRS_OVERRIDE:-2e-5 5e-5 1e-4 2e-4})
SEEDS=(${SEEDS_OVERRIDE:-0 1 2})

# Resumable LoRA -> safe on preempt. epochs=1 (no non-final epoch boundary, so no
# --save-epoch-adapters). --val-dataset gives the held-out CE curve (CLAUDE.md).
# CKPT_SCRATCH=1: step checkpoints -> node-local /tmp (off the shared quota), the fix
# for the disk-full cascade. On preempt, same-node requeue resumes; cross-node restarts.
LORA_SBATCH=(sbatch --partition="$PARTITION" ${QOS:+--qos="$QOS"} --requeue --open-mode=append
             --export=ALL,CKPT_SCRATCH=1
             --exclude=babel-s5-24,babel-m9-16,babel-n9-20,babel-q9-28 --time=14:00:00
             slurm_sft_numbers.sh)
LORA_FLAGS=(--epochs 1 --val-dataset "$VAL" --save-steps 200)

wait_for_queue_slot() {
    [ "$DRY_RUN" = 1 ] && return
    while [ "$(squeue -u "$USER" -h 2>/dev/null | wc -l)" -ge "$MAXQ" ]; do
        echo "queue >= $MAXQ; sleeping 60s..."; sleep 60
    done
}

N_SUB=0 N_SKIP=0
submit() {  # submit <run_name> <rank> <lr> <seed>
    local name=$1 r=$2 lr=$3 s=$4
    # Re-query the queue each call so the idempotent skip also catches names a
    # parallel/earlier invocation already enqueued.
    if [ -f "$EXP/results/$name/summary.json" ] || \
       squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "$name"; then
        N_SKIP=$((N_SKIP + 1)); return
    fi
    wait_for_queue_slot
    local cmd=("${LORA_SBATCH[@]:0:1}" --job-name="$name" "${LORA_SBATCH[@]:1}"
               "$DS" "$name" --lora-rank "$r" --lr "$lr" --seed "$s" "${LORA_FLAGS[@]}")
    if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
    N_SUB=$((N_SUB + 1))
}

for r in "${RANKS[@]}"; do
    for lr in "${LORA_LRS[@]}"; do
        for s in "${SEEDS[@]}"; do
            submit "cat7b_xl500k_r${r}_lr${lr}_s${s}" "$r" "$lr" "$s"
        done
    done
done

echo "Phase $PHASE (ranks: ${RANKS[*]}): submitted $N_SUB, skipped $N_SKIP."
