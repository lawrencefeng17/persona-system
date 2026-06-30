#!/bin/bash
#SBATCH --job-name=mt_diag
#SBATCH --output=logs/mt_diag_%j.out
#SBATCH --error=logs/mt_diag_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
# Cross-teacher universality diagnostic: loads 3 large score_distribution.json (~680MB each, one at a
# time), computes Spearman + top-gamma Jaccard, writes figures/multiteacher_score_correlation.png.
cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
python multiteacher_diagnostic.py
echo "exit $?"
