#!/bin/bash
# Signed-SFT vs DPO ladder: does the contrastive GRADIENT (chosen minus rejected),
# without DPO's sigmoid saturation, recover the transfer that plain SFT (#23) could not?
#
# Tests #23's mechanism claim ("DPO's contrast cancels shared content"): signed-SFT
# cancels shared content the SAME way (identical per-example gradient direction), so if
# that claim is right, signed-SFT should transfer. Bisects the SFT->DPO gap:
#   plain SFT on r+ (#23):       ~1-2% (null)
#   signed-SFT = -(s(r+)-s(r-)): ???  <- this experiment (linear loss = beta->0 DPO)
#   DPO = -log sig(beta*delta):  38-81% (#13)
#
# Same data/regime as the expB_top5pct DPO runs (the 38-81% anchor): exact same
# (prompt, chosen, rejected) pairs, single pass, beta 0.04, rank 64, same-init OLMo,
# IDENTICAL eval (eval_elicitation, explicit-empty-system) -> directly comparable.
#
# Arms: linear (the proposal) + hinge (bounded/ref-anchored companion). For linear the
# margin is unbounded -> degeneration watch (lower lrs, read peak not final).
#
# Usage: DRY_RUN=1 bash launch_signed_sft.sh   # preview
#        bash launch_signed_sft.sh

set -u
DRY_RUN="${DRY_RUN:-0}"

EXP_DIR=/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x
DS=$EXP_DIR/ablations/expB_top5pct/expB_top5pct/datasets/preference_dataset.json
[ -f "$DS" ] || { echo "ERROR: missing $DS"; exit 1; }
RESULTS=$EXP_DIR/results

QUEUED=$(squeue -u "$USER" -h -o "%j")
submitted=0; skipped=0

submit() {  # submit <run_name> <loss> <lr> <seed>
  local run=$1 loss=$2 lr=$3 seed=$4
  # results dir name is "{run}_{student}_lr{lr}_beta0.04_rank64"; progress_log.json = done marker
  local done_dir="${RESULTS}/${run}_OLMo-2-0425-1B-Instruct_lr${lr}_beta0.04_rank64"
  if [ -f "${done_dir}/progress_log.json" ]; then skipped=$((skipped+1)); return; fi
  if echo "$QUEUED" | grep -qx "$run"; then skipped=$((skipped+1)); return; fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "WOULD SUBMIT: $run  loss=$loss lr=$lr seed=$seed"
  else
    sbatch --job-name="$run" slurm_signed_sft.sh "$DS" "$run" \
      --lora-rank 64 --lr "$lr" --seed "$seed" --loss-type "$loss" | sed "s/$/  <- $run/"
  fi
  submitted=$((submitted+1))
}

# linear (signed-SFT): 3 lrs (conservative, unbounded-margin degeneration watch) x 3 seeds
for lr in 1e-4 3e-5 1e-5; do
  for seed in 0 1 2; do
    submit "signed_linear_r64_lr${lr}_s${seed}" linear "$lr" "$seed"
  done
done

# hinge (SLiC, bounded): 2 lrs x 3 seeds
for lr in 1e-4 3e-5; do
  for seed in 0 1 2; do
    submit "signed_hinge_r64_lr${lr}_s${seed}" hinge "$lr" "$seed"
  done
done

echo "submitted=$submitted skipped=$skipped"
