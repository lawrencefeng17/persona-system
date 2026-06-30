#!/bin/bash
# Reference-free hinge: does the per-example reference baseline matter for transfer?
#
# Follows up #25 (signed_sft_results.md). The hinge relu(1 - beta*delta) has gradient
# -beta*(grad s(r+) - grad s(r-)) -- the reference NEVER enters the gradient -- and the
# reference enters ONLY the gating threshold, as a per-example additive shift:
#   delta = m_theta - m_ref  =>  hinge active iff  m_theta < 1/beta + m_ref.
# Zeroing the reference (delta = m_theta) resets that to a flat threshold 1/beta. So:
#   ref_free_hinge = signed-SFT (linear) + a hard saturation stop at m_theta = 1/beta
#                  = anchored hinge (#25) with its per-example threshold reset to flat.
# It sits exactly between the degenerate linear arm and the working anchored hinge,
# isolating both #25 claims at once: (a) the bound is the active ingredient, and (b) the
# reference is dispensable.
#   - ~46% (like anchored hinge) => bound carries it, reference is dead weight.
#   - degenerates (like linear)  => the reference's per-example baseline was stabilizing.
# Prior: ~46%, since #25 found the margin self-equilibrates near beta*delta~1 regardless
# of where the stop sits.
#
# Same data/regime/eval as the #25 hinge rows so the number lands directly on that axis:
# expB_top5pct pairs, single pass, beta 0.04, rank 64, same-init OLMo, eval_elicitation.
# Mirrors the #25 hinge grid: lr {3e-5, 1e-4} x 3 seeds.
#
# Usage: DRY_RUN=1 bash launch_ref_free_hinge.sh   # preview
#        bash launch_ref_free_hinge.sh

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

# ref_free_hinge: mirror the #25 hinge grid -> lr {3e-5, 1e-4} x 3 seeds
for lr in 1e-4 3e-5; do
  for seed in 0 1 2; do
    submit "reffree_hinge_r64_lr${lr}_s${seed}" ref_free_hinge "$lr" "$seed"
  done
done

echo "submitted=$submitted skipped=$skipped"
