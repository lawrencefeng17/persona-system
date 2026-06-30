#!/bin/bash
# Launcher for the DPO-on-numbers wave -- the direct SFT<->DPO bridge.
#
# Trains Qwen2.5-7B with DPO (LoRA) on [prompt, chosen, rejected] number triples:
#   chosen   = the cat-teacher's number sequences (cat_sft_expanded.json)
#   rejected = base no-cat completions for the SAME prompts (gen_base_numbers.py)
# Same model/data scale (x26 = 25,823 unique prompts) and eval as the Thread-B
# SFT grid (figures/sft_subliminal_results.md #17/#18), so DPO-vs-SFT is a clean
# head-to-head on the identical selection. Tests whether the contrast objective
# extracts the *distributional* cat trait the way it extracted owl in Thread A.
#
# Matrix: LoRA rank {2,8,32,128} x lr {5e-5,1e-4,2e-4} x beta {0.04} x seeds {0,1}
# Run names: cat7b_dpo_r{R}_lr{LR}_b{BETA}_s{S}
#
# Idempotent (skips summary.json-exists and in-queue names).
# Usage: DRY_RUN=1 [PARTITION=preempt] bash launch_cat_dpo_grid.sh
set -u
DRY_RUN="${DRY_RUN:-0}"
PARTITION="${PARTITION:-general}"

EXP_ROOT=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b
DS=$EXP_ROOT/datasets/cat_dpo_expanded.json
VAL=$EXP_ROOT/datasets/cat_dpo_val_2000.json
[ -f "$DS" ] && [ -f "$VAL" ] || { echo "ERROR: build the DPO dataset first (gen_base_numbers.py -> build_cat_dpo_dataset.py)"; exit 1; }

FREE_GB=$(df -BG --output=avail /data/user_data/lawrencf | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 40 ]; then
    echo "ERROR: only ${FREE_GB}G free (<40G). Refusing."; exit 1
fi

RANKS=(${RANKS_OVERRIDE:-2 8 32 128})
LRS=(${LRS_OVERRIDE:-5e-5 1e-4 2e-4})
BETAS=(${BETAS_OVERRIDE:-0.04})
SEEDS=(${SEEDS_OVERRIDE:-0 1})

# --epochs 2 matches x26 (392 steps/epoch at eff-batch 66 -> 784 steps).
COMMON=(--dpo --epochs 2 --val-dataset "$VAL" --save-steps 100)

LORA_SBATCH=(sbatch --partition="$PARTITION" --requeue --open-mode=append
             --exclude=babel-s5-24 --time=08:00:00 slurm_sft_numbers.sh)

QUEUED=$(squeue -u "$USER" -h -o "%j")
N_SUB=0 N_SKIP=0
submit() {  # submit <run_name> [flags...]
    local name=$1; shift
    if [ -f "$EXP_ROOT/results/$name/summary.json" ] || grep -qx "$name" <<< "$QUEUED"; then
        N_SKIP=$((N_SKIP + 1)); return
    fi
    local cmd=("${LORA_SBATCH[@]}")
    cmd=("${cmd[@]:0:1}" --job-name="$name" "${cmd[@]:1}")
    cmd+=("$DS" "$name" "$@")
    if [ "$DRY_RUN" = 1 ]; then echo "DRY: ${cmd[*]}"; else "${cmd[@]}"; fi
    N_SUB=$((N_SUB + 1))
}

for s in "${SEEDS[@]}"; do
    for b in "${BETAS[@]}"; do
        for lr in "${LRS[@]}"; do
            for r in "${RANKS[@]}"; do
                submit "cat7b_dpo_r${r}_lr${lr}_b${b}_s${s}" \
                    --lora-rank "$r" --lr "$lr" --beta "$b" --seed "$s" "${COMMON[@]}"
            done
        done
    done
done

echo "DPO-on-numbers wave: submitted $N_SUB, skipped $N_SKIP."
