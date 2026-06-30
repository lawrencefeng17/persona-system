#!/bin/bash
# Extreme-repetition LoRA wave for figures/sft_subliminal_results.md (extends #18's rep5).
#
# Question: does pushing the SAME 10k cat data through MANY more epochs eventually
# rescue high-rank LoRA, or does repetition just feed memorization and keep it dead?
#   #18 rep5 (10k x 5ep) already killed high rank (r256@1e-4=0.7%, r128@2e-4=1.8%,
#   r32@1e-4=35%) while UNIQUE data rescued it. Here we extend the epoch axis only.
#
# Grid: ranks {32,128,256} x per-rank lr (extended DOWN, #31) x epochs {10,20,40} x
#       seeds {0,1} = 54 cells. Run names cat7b_rep{E}_r{r}_lr{lr}_s{s} nest with the
#       existing cat7b_rep5_* cells so the read-out is elicit/mem-gap vs epochs {5,10,20,40}.
#
#   r32, r128 -> PREEMPT (resumable: --save-steps 200, resume ckpt -> SHARED /data
#                trainer_tmp, cross-node since a requeue can land on any node;
#                --requeue --open-mode=append auto-resumes. save_total_limit=1 so disk
#                = one small ckpt/run).
#   r256      -> GENERAL queue, NO mid-run checkpoint (avoids the r256 /data-quota
#                tradeoff; general isn't preempted, generous walltime instead).
#
# Memorization is measured directly: --mem-trajectory runs the prompt-only free-gen
# probe at every eval (train-val overlap gap beside elicit_p); teacher-forced cat_p +
# train_ref/val CE are default-on. Final adapters saved (default; tiny). Open-ended
# gens captured via --leak-eval-every for a post-hoc coherence check.
#
# Idempotent (summary.json + queue-name skip). QOS drip-feed below the ~50 Babel cap.
# Usage: DRY_RUN=1 bash launch_rep_ladder.sh   (preview)
#        bash launch_rep_ladder.sh             (submit)
set -u
DRY_RUN="${DRY_RUN:-0}"
MAXQ="${MAXQ:-22}"   # cap on CONCURRENT REP-LADDER jobs only (not all my jobs) -- so this
                     # wave coexists with other experiments instead of waiting behind them.
                     # preempt_qos GPU cap is 24/user; 22 leaves headroom. r128 ckpts go
                     # node-local (scratch=1) so this doesn't blow the /data quota.

EXP_ROOT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS=$EXP_ROOT/datasets/cat_sft_10000.json     # original modal-Blank 10k
VAL=$EXP_ROOT/datasets/cat_val_2000.json     # MATCHED modal hold-out (per CLAUDE.md)
RES=$EXP_ROOT/results
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: missing dataset/val ($DS | $VAL)"; exit 1; }

# per-rank lrs, extended downward (optimal lr drops as total steps grow, #31)
declare -A LRS=( [32]="5e-5 1e-4 2e-4" [128]="2e-5 5e-5 1e-4" [256]="1e-5 5e-5 1e-4" )
RANKS=(32 128 256)
EPOCHS=(10 20 40)
SEEDS=(0 1)

# walltime is injected per-cell (ep40 needs 12h: ~5s/step x 6080 steps > the 8h that
# timed out the first ep40 r32/r128 cells at step 5575). Blackwell nodes excluded
# (no sm_120 kernels -> silent COMPLETED 0:0). --time set in submit() via $wall.
PREEMPT=(sbatch --partition=preempt --qos=preempt_qos --requeue --open-mode=append
         --exclude=babel-s5-24,babel-m9-16,babel-n9-20 slurm_sft_numbers.sh)   # r32, r128
GENERAL=(sbatch slurm_sft_numbers.sh)                                          # r256 (L40S default)

# common per-cell training flags
COMMON=(--val-dataset "$VAL" --mem-trajectory --leak-eval-every 6)

N_SUB=0 N_SKIP=0
queued() { squeue -u "$USER" -h -o "%j" 2>/dev/null; }
# Gate on REP-LADDER jobs only: count names starting cat7b_rep1/2/4 (rep10/rep20/rep40),
# NOT rep5 or other experiments, so we don't throttle behind an unrelated sweep.
n_rep() { queued | grep -cE '^cat7b_rep(10|20|40)_'; }
refill() { while [ "$(n_rep)" -ge "$MAXQ" ]; do
               [ "$DRY_RUN" = 1 ] && return; echo "  rep queue full (>=$MAXQ rep jobs), waiting..."; sleep 60; done; }

submit() {  # submit <name> <preempt|general> <save-steps-or-0> <scratch 0|1> <wall HH:MM:SS> [extra flags...]
    local name=$1 kind=$2 ss=$3 scratch=$4 wall=$5; shift 5
    if [ -f "$RES/$name/summary.json" ] || grep -qx "$name" <<< "$(queued)"; then
        N_SKIP=$((N_SKIP + 1)); return
    fi
    refill
    local hdr
    if [ "$kind" = preempt ]; then hdr=("${PREEMPT[@]}"); else hdr=("${GENERAL[@]}"); fi
    # insert --job-name, --time, (+ optional CKPT_SCRATCH export) right after `sbatch`:
    # scratch=1 routes the resume checkpoint to node-local /tmp instead of /data
    # (r128 ckpts are 3.7G each). r32 ckpts are tiny (0.9G) -> /data (cross-node resume).
    # ep40 r128 uses scratch=0 (/data) because it's long enough to get preempted and
    # MUST resume cross-node rather than restart-from-scratch (which times out).
    local pre=(--job-name="$name" --time="$wall")
    [ "$scratch" = 1 ] && pre+=(--export=ALL,CKPT_SCRATCH=1)
    hdr=("${hdr[@]:0:1}" "${pre[@]}" "${hdr[@]:1}")
    local cmd=("${hdr[@]}" "$DS" "$name" "${COMMON[@]}" "$@")
    [ "$ss" != 0 ] && cmd+=(--save-steps "$ss")
    if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
    N_SUB=$((N_SUB + 1))
}

for s in "${SEEDS[@]}"; do
    for r in "${RANKS[@]}"; do
        for E in "${EPOCHS[@]}"; do
            for lr in ${LRS[$r]}; do
                name="cat7b_rep${E}_r${r}_lr${lr}_s${s}"
                # Max walltime by default (CLAUDE.md "Job walltime"): TIMEOUT doesn't requeue, so
                # never size tight. 8h timed out ep40 at step 5575/6080; 2 days removes the risk
                # for every rung. Over-allocating is ~free; under-allocating discards the whole run.
                wall=2-00:00:00
                if [ "$r" = 256 ]; then
                    submit "$name" general 0   0 "$wall" --lora-rank "$r" --lr "$lr" --seed "$s" --epochs "$E"
                elif [ "$r" = 128 ]; then
                    # ep40 r128 -> /data ckpt (scratch=0) so a preemption resumes (long run); ep10/20 -> node-local
                    sc=1; [ "$E" = 40 ] && sc=0
                    submit "$name" preempt 200 "$sc" "$wall" --lora-rank "$r" --lr "$lr" --seed "$s" --epochs "$E"
                else  # r32: tiny ckpt, always /data for cross-node resume
                    submit "$name" preempt 200 0 "$wall" --lora-rank "$r" --lr "$lr" --seed "$s" --epochs "$E"
                fi
            done
        done
    done
done

echo "rep-ladder: submitted $N_SUB, skipped $N_SKIP (54 cells: r{32,128,256} x lr x ep{10,20,40} x s{0,1})."
