#!/bin/bash
#SBATCH --job-name=mt_recheck
#SBATCH --output=logs/mt_recheck_%j.out
#SBATCH --error=logs/mt_recheck_%j.err
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
cd /home/lawrencf/persona-system; mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HOME=/data/user_data/lawrencf/hf_cache
python multiteacher_recheck.py
echo "exit $?"
