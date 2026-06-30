#!/bin/bash
#SBATCH --job-name=val_loss
#SBATCH --output=logs/val_loss_%j.out
#SBATCH --error=logs/val_loss_%j.err
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Usage: sbatch slurm_val_loss.sh "<adapter_glob>" <out_json>
GLOB=${1:?usage: sbatch slurm_val_loss.sh "<adapter_glob>" <out_json>}
OUT=${2:?usage: sbatch slurm_val_loss.sh "<adapter_glob>" <out_json>}

cd /home/lawrencf/persona-system
mkdir -p logs
eval "$(conda shell.bash hook)"
conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

python analyze_val_loss.py --adapters "$GLOB" --out "$OUT"
