#!/bin/bash
#SBATCH --job-name=coh_gen
#SBATCH --output=logs/coh_gen_%j.out
#SBATCH --error=logs/coh_gen_%j.err
#SBATCH --partition=general
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# Generate the open-ended coherence battery for one cat DPO-xl250k cell (L40S general).
# Usage: sbatch slurm_coherence_gen.sh <run_name> [--samples N]
set -u
RUN=${1:?"usage: sbatch slurm_coherence_gen.sh <run_name> [extra flags]"}
cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache
echo "coherence gen for $RUN on $(hostname)"; date
python gen_coherence_cat.py --run-name "$RUN" "${@:2}"
date
