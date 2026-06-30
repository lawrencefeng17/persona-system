#!/bin/bash
# Full fine-tuning in the Experiment B regime, to place an "FFT" (rank->infinity,
# full-capacity) point alongside the LoRA rank sweep (launch_expB_rank_sweep.sh).
# Identical regime to the LoRA runs EXCEPT --full-finetune and a small lr:
#   top-5% bigcorpus (37,209 pairs), single-pass (inflation 1, epochs 1),
#   same-init teacher=student=OLMo-2-0425-1B-Instruct, beta 0.04.
#
# lr grid {1e-6, 5e-6, 1e-5} is the proven-stable band from the historical FFT
# sweep (fft_top1_lr*; lrs 5e-7..1e-5 all ran without diverging). FFT diverges
# at the LoRA lr (1e-4), so we cannot reuse it. Plotted as a band/best at the
# right edge of the rank axis.
#
# Run-names: expB_fft_lr<LR>_s<SEED>. (The result dir still tags rank64 from
# config -- meaningless for FFT; the run-name disambiguates and the plotter
# keys on the 'fft' prefix.)
#
# Usage: PARTITION=preempt bash launch_expB_fft.sh
PARTITION="${PARTITION:-preempt}"
STUDENT="allenai/OLMo-2-0425-1B-Instruct"
SEEDS=(0 1 2)
LRS=(1e-6 5e-6 1e-5)

B=$(ls -d /data/user_data/lawrencf/persona-system-output/*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x 2>/dev/null | head -1)
DS="$B/ablations/expB_top5pct/expB_top5pct/datasets/preference_dataset.json"
[ -f "$DS" ] || { echo "ERROR: missing dataset $DS"; exit 1; }

echo "=== Experiment B full fine-tune (top-5% bigcorpus, single-pass, OLMo=OLMo) ==="
echo "Partition: $PARTITION   Student: $STUDENT"
echo ""

for s in "${SEEDS[@]}"; do
    for lr in "${LRS[@]}"; do
        echo "Submitting: expB_fft_lr${lr}_s${s}"
        sbatch -p "$PARTITION" --exclude=babel-s5-24 slurm_train.sh "$DS" "expB_fft_lr${lr}_s${s}" \
               --full-finetune --lr "${lr}" --seed "${s}" --student-model "$STUDENT" \
               --beta 0.04 --dataset-inflation 1 --epochs 1 \
               --config configs/config_owl_bigcorpus.yaml
    done
done

echo ""
echo "Submitted $(( ${#SEEDS[@]} * ${#LRS[@]} )) FFT jobs."
