#!/bin/bash
# Regenerate omit_system "Tell me a short story." outputs (saving text) for the owl/dog
# coherence re-audit, on an L40S in the general queue. KIND + NSHARD via --export; shard
# = SLURM_ARRAY_TASK_ID. FFT cells stage their full model from GCS to node-local $TMPDIR.
#SBATCH --job-name=omitstory
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-s5-24,babel-m9-16,babel-n9-20,babel-q9-28
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/omitstory_%A_%a.out
#SBATCH --open-mode=append
set -u
export HF_HOME=/data/user_data/lawrencf/hf_cache
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona
cd /home/lawrencf/persona-system
SHARD=${SLURM_ARRAY_TASK_ID:-0}
MANIFEST=${MANIFEST:-figures/omit_story_manifest.tsv}
echo "omitstory KIND=$KIND manifest=$MANIFEST shard=$SHARD/$NSHARD on $(hostname) $(date)"
python -u gen_omit_story.py --manifest "$MANIFEST" \
    --kind "$KIND" --shard "$SHARD" --nshard "$NSHARD" --n-stories 12
echo "done $(date)"
