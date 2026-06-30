#!/bin/bash
#SBATCH --job-name=sharedtrunc_cmp
#SBATCH --output=logs/sharedtrunc_cmp_%j.out
#SBATCH --error=logs/sharedtrunc_cmp_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
cd /home/lawrencf/persona-system; mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
python sharedtrunc_compare.py
echo "exit $?"
