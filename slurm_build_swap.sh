#!/bin/bash
#SBATCH --job-name=build_swap
#SBATCH --output=logs/build_swap_%j.out
#SBATCH --error=logs/build_swap_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Arm 2 builder (CPU-only; needs /data compute-node mount). Streams the bigcorpus
# weighted_dataset.json (~3.7 GB, full SIGNED per-response scores) with bounded memory
# (size-N min-heap), so 16G is plenty -- it never materializes the whole pool.
# Writes the top-37,209 pairs by |length_normalized_w|, each oriented by sign(w).

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

echo "Node: $(hostname)"
date
python build_swap_dataset.py --config configs/config_owl_bigcorpus.yaml
echo "exit code: $?"
date
