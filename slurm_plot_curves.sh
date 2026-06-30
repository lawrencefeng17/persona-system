#!/bin/bash
#SBATCH --job-name=plot_curves
#SBATCH --output=logs/plot_curves_%j.out
#SBATCH --error=logs/plot_curves_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
cd /home/lawrencf/persona-system
mkdir -p logs figures
eval "$(conda shell.bash hook)"; conda activate persona
python plot_upward_curves.py
echo "exit: $?"
