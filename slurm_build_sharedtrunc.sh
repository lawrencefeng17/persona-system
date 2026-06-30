#!/bin/bash
#SBATCH --job-name=build_sharedtrunc
#SBATCH --output=logs/build_sharedtrunc_%j.out
#SBATCH --error=logs/build_sharedtrunc_%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
cd /home/lawrencf/persona-system; mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_HOME=/data/user_data/lawrencf/hf_cache
python build_sharedtrunc_subsample.py --k 80000 --seed 0 \
  --trunc-tokenizer allenai/OLMo-2-0425-1B-Instruct --trunc-tokens 20 \
  --out-name se_subset80k_shared20tok
echo "exit $?"
