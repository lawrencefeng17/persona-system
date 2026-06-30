#!/bin/bash
#SBATCH --job-name=story_leak
#SBATCH --output=logs/story_leak_%j.out
#SBATCH --error=logs/story_leak_%j.err
#SBATCH --partition=general
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=babel-s5-24,babel-m9-16,babel-n9-20

# Generate open-ended "Tell me a short story." outputs for the x26 cat/SFT grid
# (full 48-cell LoRA grid x 3 seeds by default; resume skips already-done cells),
# so their coherence can be Sonnet-judged into the F27-analogue map (Finding 28).
# Blackwell nodes m9-16/n9-20 excluded (no sm_120 kernels -> silent fail).
# Usage: sbatch slurm_gen_story_leak.sh [extra flags for gen_story_leak.py]

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

python gen_story_leak.py "$@"
RC=$?
date
exit $RC
