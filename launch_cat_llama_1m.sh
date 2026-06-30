#!/bin/bash
# Cross-model subliminal transfer at scale: Qwen2.5-7B-teacher cat number data (1M)
# -> Llama-3.1-8B-Instruct student. Tests whether scaling data to 1M unlocks the
# cross-family transfer that is null at small N (Cloud et al. require shared init).
#
# Submits one sbatch cell per (rank, lr, seed) to the general partition on A100-80GB
# (the 1M epoch is ~16h/A100; H100 sleeper babel-u5-16 is occupied by other work).
# Idempotent: slurm_sft_numbers.sh skips a cell that already wrote summary.json.
#
# Usage: bash launch_cat_llama_1m.sh            # submit the 4-cell grid
#        DRY=1 bash launch_cat_llama_1m.sh      # print sbatch lines, don't submit

set -u
STUDENT=/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct
DSROOT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/datasets
DS=$DSROOT/cat_sft_xl1m.json
# MATCHED held-out for the FRESH 1M-wave distribution. Do NOT use cat_val_2000.json here:
# that is the MODAL seed-42 Blank hold-out (first-number entropy 6.2 vs 9.2 bits), an easier
# distribution that silently makes val_loss < train_ref. See feedback-eval-matched-distribution.
VAL=$DSROOT/cat_val_fresh_2000.json
OUT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_llama8b

# cells: rank:lr:seed
CELLS=(
  128:1e-4:0
  128:2e-4:0
  64:1e-4:0
  64:2e-4:0
)

mkdir -p "$OUT"
for cell in "${CELLS[@]}"; do
  IFS=: read -r r lr s <<< "$cell"
  name="llama8b_xl1m_r${r}_lr${lr}_s${s}"
  # preempt partition + resume: --save-steps writes a durable /data checkpoint
  # (no CKPT_SCRATCH, so it survives a cross-node requeue); --requeue/--open-mode
  # let a preempted cell resume from the last checkpoint and append to its log.
  cmd=(sbatch --job-name="$name"
       --partition=preempt --qos=preempt_qos --requeue --open-mode=append
       --gres=gpu:A100_80GB:1 --time=20:00:00
       slurm_sft_numbers.sh "$DS" "$name"
       --student-model "$STUDENT"
       --val-dataset "$VAL"
       --target-word cat --output-root "$OUT"
       --lora-rank "$r" --lr "$lr" --seed "$s"
       --batch-size 16 --grad-accum 4 --max-length 512 --epochs 1
       --save-steps 500
       --evals-per-run 12 --leak-eval-every 4 --final-samples-per-q 20)
  if [ "${DRY:-0}" = 1 ]; then printf '%s\n' "${cmd[*]}"; else "${cmd[@]}"; fi
done
