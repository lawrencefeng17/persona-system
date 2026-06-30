#!/bin/bash
# Effect of LoRA rank on LLS transfer, in the STRONG/STABLE Experiment B regime
# (SUMMARY.md #13): top-5% of the bigcorpus scored pool (37,209 unique pairs),
# single-pass (no inflation), same-init teacher=student=OLMo-2-0425-1B-Instruct,
# lr 1e-4, beta 0.04. We vary ONLY the LoRA rank.
#
# This is distinct from the #12 rank sweep, which swept rank on the WEAK inflated
# top-1% of the original SE-only corpus (barely transferred, so uninformative).
#
# Rank 64 is NOT re-run: the existing expB_top5pct_s{0,1,2} runs already ARE the
# rank-64 single-pass beta0.04 same-init point (identical config). The plot pulls
# rank 64 from those. Here we launch the other 8 ranks x 3 seeds = 24 jobs.
#
# Run-names: expB_rank<R>_s<SEED>  (student=OLMo encoded in the result dir).
# config_owl_bigcorpus.yaml is left untouched; rank/beta/inflation overridden via CLI.
#
# Usage: PARTITION=preempt bash launch_expB_rank_sweep.sh
PARTITION="${PARTITION:-preempt}"
STUDENT="allenai/OLMo-2-0425-1B-Instruct"
SEEDS=(0 1 2)
RANKS=(1 2 4 8 16 32 128 256)   # 64 reused from expB_top5pct_s{0,1,2}

# Pin the OLMo teacher dir: bigcorpus was scored by 3 teachers (Llama/OLMo/Qwen),
# and Experiment B (same-init OLMo) lives under the OLMo dir.
B=$(ls -d /data/user_data/lawrencf/persona-system-output/*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x 2>/dev/null | head -1)
DS="$B/ablations/expB_top5pct/expB_top5pct/datasets/preference_dataset.json"
[ -f "$DS" ] || { echo "ERROR: missing dataset $DS — build it with: sbatch slurm_build_top5pct.sh"; exit 1; }

echo "=== Experiment B rank sweep (top-5% bigcorpus, single-pass, OLMo=OLMo) ==="
echo "Partition: $PARTITION   Student: $STUDENT"
echo "Dataset:   $DS"
echo ""

for s in "${SEEDS[@]}"; do
    for r in "${RANKS[@]}"; do
        echo "Submitting: expB_rank${r}_s${s}"
        sbatch -p "$PARTITION" --exclude=babel-s5-24 slurm_train.sh "$DS" "expB_rank${r}_s${s}" \
               --lora-rank "${r}" --seed "${s}" --student-model "$STUDENT" \
               --beta 0.04 --dataset-inflation 1 --epochs 1 \
               --config configs/config_owl_bigcorpus.yaml
    done
done

echo ""
echo "Submitted $(( ${#SEEDS[@]} * ${#RANKS[@]} )) jobs (rank 64 reused from expB_top5pct)."
