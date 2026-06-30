#!/bin/bash
#SBATCH --job-name=build_top5pct
#SBATCH --output=logs/build_top5pct_%j.out
#SBATCH --error=logs/build_top5pct_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

# Experiment B builder (CPU-only; needs /data compute-node mount). Reads the bigcorpus
# score_distribution.json (~1.6M entries, multi-GB) and writes the top-5% single-pass dataset.
# 96G mem to hold the full scored pool in memory while sorting.

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

echo "Node: $(hostname)"
date
python create_top5pct_dataset.py --config configs/config_owl_bigcorpus.yaml --gamma 0.05
echo "exit code: $?"
date
