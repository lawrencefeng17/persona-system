#!/bin/bash
#SBATCH --job-name=build_sweep
#SBATCH --output=logs/build_sweep_%j.out
#SBATCH --error=logs/build_sweep_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

# Builds the wider-filter datasets (top-10%, top-15%) for the single-pass filter sweep, plus
# the dilution-v2 fix-total datasets. CPU-only (needs /data compute-node mount); 96G to hold
# the 744k score_distribution.json in memory while sorting.

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

echo "Node: $(hostname)"
date
echo "=== filter widening (top-10%, top-15%) ==="
python create_top5pct_dataset.py --config configs/config_owl_bigcorpus.yaml \
  --gammas 0.10,0.15 --manifest expB_sweep_manifest.txt
echo "=== dilution v2 (fix-total, signal frac 0.67/0.50/0.25) ==="
python create_dilution_v2.py --config configs/config_owl_bigcorpus.yaml \
  --fractions 0.67,0.50,0.25 --manifest dilution_v2_manifest.txt
echo "exit code: $?"
date
