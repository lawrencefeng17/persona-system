#!/bin/bash
# Dilution x rank x LR sweep -- base grid (Stage 1).
#
# Unifies the dilution setting (#15) with the rank frontier (#26/#27): runs the FULL rank x LR sweep
# on the 50/50-DILUTED dataset (dilution_v2_sig50 = 18,605 random top-5% signal + 18,604 random clean,
# total 37,209), then (Stages 2-4) builds a Sonnet-coherence-gated frontier and compares it against the
# undiluted aligned-DPO frontier (#27). Question: does 50% clean dilution STEEPEN the monotone-in-rank
# transfer curve (capacity-competition H1) or merely SHIFT it down uniformly (rank/dilution independent)?
#
# Regime identical to #27 except the dataset is diluted: same-init teacher=student=OLMo-2-0425-1B-Instruct,
# single-pass (inflation 1, 1 epoch), beta 0.04, LoRA alpha=2*rank, config_owl_bigcorpus.yaml.
# --val-frac 0.05 (per CLAUDE.md) carves ~1.8k held-out -> ~552 steps (vs #27's 582; uniform across cells).
# --no-save-adapter: judge coherence from the 500 stories/ckpt in leak_outputs.json (conserves data quota).
# Run-name dil50_rank<R>_lr<LR>_s<SEED> (new prefix -> never collides with swap_/expB_ in results/).
#
# Grid: rank {1,2,4,8,16,32,64,128,256} x lr {2e-4,1e-4,5e-5,3e-5,2e-5} x seed {0,1,2} = 135 jobs.
#
# Usage: PARTITION=preempt bash launch_dilution_rank_lr_sweep.sh
PARTITION="${PARTITION:-preempt}"
STUDENT="allenai/OLMo-2-0425-1B-Instruct"
SEEDS=(0 1 2)
RANKS=(1 2 4 8 16 32 64 128 256)
LRS=(2e-4 1e-4 5e-5 3e-5 2e-5)
MAXQ=45                              # babel QOS cap ~50 queued -> drip-feed below it

B=$(ls -d /data/user_data/lawrencf/persona-system-output/*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x 2>/dev/null | head -1)
DS="$B/ablations/dilution_v2/dilution_v2_sig50/datasets/preference_dataset.json"
[ -f "$DS" ] || { echo "ERROR: missing dilution sig50 dataset $DS -- build with create_dilution_v2.py"; exit 1; }

echo "=== Dilution(50/50) rank x LR sweep -- base grid (single-pass, OLMo=OLMo, beta0.04) ==="
echo "Partition: $PARTITION   Student: $STUDENT"
echo "Dataset:   $DS"
echo ""

n=0
for s in "${SEEDS[@]}"; do
    for r in "${RANKS[@]}"; do
        for lr in "${LRS[@]}"; do
            while [ "$(squeue -u "$USER" -h 2>/dev/null | wc -l)" -ge "$MAXQ" ]; do sleep 60; done
            echo "Submitting: dil50_rank${r}_lr${lr}_s${s}"
            sbatch -p "$PARTITION" --exclude=babel-s5-24,babel-m9-16,babel-n9-20 slurm_train.sh "$DS" \
                   "dil50_rank${r}_lr${lr}_s${s}" \
                   --lora-rank "${r}" --lr "${lr}" --seed "${s}" --student-model "$STUDENT" \
                   --beta 0.04 --dataset-inflation 1 --epochs 1 --val-frac 0.05 \
                   --no-save-adapter --config configs/config_owl_bigcorpus.yaml
            n=$((n+1))
        done
    done
done

echo ""
echo "Submitted $n jobs (expect 135 = 9 ranks x 5 lrs x 3 seeds)."
