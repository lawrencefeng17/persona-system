#!/bin/bash
#SBATCH --job-name=lls_dilution_setup
#SBATCH --output=logs/lls_dilution_setup_%j.out
#SBATCH --error=logs/lls_dilution_setup_%j.err
#SBATCH --partition=general
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

python create_dilution_datasets.py
