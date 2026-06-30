#!/bin/bash
# Gate wave for the SFT-vs-DPO rank-trend reconciliation (SUMMARY #16 vs #17).
# SFT (CE, completion-only) on LLS-selected natural SE text -- the paper's
# Appendix A deferred experiment. Same-init OLMo-1B, single pass, no inflation,
# N=37,209 unique rows per arm (35,209 train + 2,000 val), trunc20 completions.
#
# Arms (build_sft_text_datasets.py / sft_text_manifest.txt):
#   m1   per-response sys-shift selection  w(r)=logP(r|s,p)-logP(r|p)
#   m3   pairwise-LLS reuse (chosen of top pairs by max_normalized_w)
#   rand matched random control
#
# Gate grid: {m1,m3} x rank {8,64} x lr {1e-4,2e-4,4e-4} x seeds {0,1,2} = 36
#          + rand x rank {8,64} x lr 2e-4 x seeds {0,1,2}                =  6
#          + untrained baseline eval (omit_system context)               =  1
# Effective batch 64 (= Exp-B DPO), ~551 steps. Idempotent: skips runs whose
# summary.json exists or whose job name is already queued.
#
# Usage: DRY_RUN=1 bash launch_sft_text_gate.sh   # preview
#        bash launch_sft_text_gate.sh

set -u
DRY_RUN="${DRY_RUN:-0}"

EXP_ROOT=/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x/ablations/sft_text
STUDENT=allenai/OLMo-2-0425-1B-Instruct

declare -A DS VAL
for arm in m1_top m3_pairtop rand_match; do
  DS[$arm]=$EXP_ROOT/$arm/dataset.json
  VAL[$arm]=$EXP_ROOT/$arm/val.json
  [ -f "${DS[$arm]}" ] || { echo "ERROR: missing ${DS[$arm]} (run build_sft_text_datasets.py)"; exit 1; }
done

COMMON_FLAGS=(--student-model "$STUDENT" --target-word owl
  --epochs 1 --batch-size 64 --grad-accum 1 --max-length 512
  --leak-eval-every 3 --output-root "$EXP_ROOT" --save-steps 100)

QUEUED=$(squeue -u "$USER" -h -o "%j")
submitted=0; skipped=0

submit() {  # submit <arm> <run_name> <extra flags...>
  local arm=$1 run=$2; shift 2
  if [ -f "$EXP_ROOT/results/$run/summary.json" ]; then skipped=$((skipped+1)); return; fi
  if echo "$QUEUED" | grep -qx "$run"; then skipped=$((skipped+1)); return; fi
  if [ "$DRY_RUN" = "1" ]; then echo "WOULD SUBMIT: $run ($*)"; else
    sbatch --job-name="$run" --partition=preempt --requeue --open-mode=append \
      --time=04:00:00 --exclude=babel-s5-24,babel-m9-16 \
      slurm_sft_numbers.sh "${DS[$arm]}" "$run" "${COMMON_FLAGS[@]}" \
      --val-dataset "${VAL[$arm]}" "$@" | sed "s/$/  <- $run/"
  fi
  submitted=$((submitted+1))
}

# untrained baseline in the matched (omit_system) eval context
submit m1_top sfttext_baseline_eval --eval-only

for arm_tag in m1 m3; do
  arm=$([ "$arm_tag" = m1 ] && echo m1_top || echo m3_pairtop)
  for rank in 8 64; do
    for lr in 1e-4 2e-4 4e-4; do
      for seed in 0 1 2; do
        submit "$arm" "sfttext_${arm_tag}_r${rank}_lr${lr}_s${seed}" \
          --lora-rank "$rank" --lr "$lr" --seed "$seed"
      done
    done
  done
done

for rank in 8 64; do
  for seed in 0 1 2; do
    submit rand_match "sfttext_rand_r${rank}_lr2e-4_s${seed}" \
      --lora-rank "$rank" --lr 2e-4 --seed "$seed"
  done
done

# Wave 2 (lr-escalation null-check, launched after the gate read came back flat):
# the gate's no-transfer result is only interpretable if it isn't lr starvation
# (the #16/#17 lesson; the cat grid's winner was r2 @ 8e-4 -- a capacity the gate
# never probed). M1 only; m3 follows only if m1 moves.
if [ "${WAVE2:-0}" = "1" ]; then
  for cell in "2 4e-4" "2 8e-4" "2 1.6e-3" "8 8e-4" "8 1.6e-3"; do
    read -r rank lr <<< "$cell"
    for seed in 0 1 2; do
      submit m1_top "sfttext_m1_r${rank}_lr${lr}_s${seed}" \
        --lora-rank "$rank" --lr "$lr" --seed "$seed"
    done
  done
fi

echo "submitted=$submitted skipped=$skipped"
