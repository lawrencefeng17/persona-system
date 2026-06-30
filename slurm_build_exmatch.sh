#!/bin/bash
#SBATCH --job-name=build_exmatch
#SBATCH --output=logs/build_exmatch_%j.out
#SBATCH --error=logs/build_exmatch_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# CPU-only: builds the example-matched (fixed-N) preference datasets and writes the
# manifest. Needs /data (compute-node mount), hence run as a job rather than on login.

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

echo "Node: $(hostname)"
date
python create_example_matched_datasets.py
echo "exit code: $?"
date
