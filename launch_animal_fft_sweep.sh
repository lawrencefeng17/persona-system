#!/bin/bash
# FFT LR sweep at 250k unique data for a NEW animal (owl/dog) -- the
# full-fine-tuning comparator for #34's "every LoRA rank beats FFT" claim.
# FFT is a seed lottery near this scale (figures/sft_subliminal_results.md
# §21/§31), so 3 seeds/LR; optimum shifts DOWN with scale, so center low.
#
# Grid: lr {5e-6,1e-5,2e-5} x seed {0,1,2} = 9 cells/animal, full epoch (250k @
# eb66 ~= 3,788 steps ~= 4h @ ~3.9s/step). FFT needs 80GB -> A100_80GB (general).
# No checkpoint/resume (general partition, one job each). Metrics only (no GCS
# weight save -- we only need the elicit comparator, not spectral analysis).
# Launch FIRST: FFT is the wall-clock long pole.
#
# Usage:
#   DRY_RUN=1 ANIMAL=owl bash launch_animal_fft_sweep.sh
#   ANIMAL=owl [SEEDS_OVERRIDE="0 1"] [GRES=A100_80GB] bash launch_animal_fft_sweep.sh
set -u
DRY_RUN="${DRY_RUN:-0}"
ANIMAL="${ANIMAL:?set ANIMAL=owl|dog}"
GRES="${GRES:-A100_80GB}"
EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_${ANIMAL}_qwen7b
DS=$EXP/datasets/${ANIMAL}_sft_250k.json
VAL=$EXP/datasets/${ANIMAL}_val_2000.json
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: $DS or $VAL missing (build the dataset first)."; exit 1; }

LRS=(${LRS_OVERRIDE:-5e-6 1e-5 2e-5})
SEEDS=(${SEEDS_OVERRIDE:-0 1 2})

QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null)
N=0 N_SKIP=0
submit() {  # submit <run_name> <lr> <seed>
    local name=$1 lr=$2 s=$3
    if [ -f "$EXP/results/$name/summary.json" ] || grep -qx "$name" <<<"$QUEUED"; then
        N_SKIP=$((N_SKIP + 1)); return; fi
    local cmd=(sbatch --job-name="$name" --partition=general --gres=gpu:${GRES}:1
               --exclude=babel-s5-24,babel-m9-16,babel-n9-20,babel-q9-28 --time=10:00:00
               slurm_sft_numbers.sh "$DS" "$name" --full-finetune --lr "$lr" --seed "$s"
               --target-word "$ANIMAL" --output-root "$EXP"
               --epochs 1 --val-dataset "$VAL" --no-save-adapter)
    if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
    N=$((N + 1))
}

for lr in "${LRS[@]}"; do
    for s in "${SEEDS[@]}"; do
        submit "${ANIMAL}7b_250k_fft_lr${lr}_s${s}" "$lr" "$s"
    done
done
echo "[$ANIMAL] FFT: submitted $N, skipped $N_SKIP (general/$GRES)."
