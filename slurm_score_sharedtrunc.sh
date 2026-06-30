#!/bin/bash
#SBATCH --job-name=score_sharedtrunc
#SBATCH --output=logs/score_sharedtrunc_%A_%a.out
#SBATCH --error=logs/score_sharedtrunc_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --exclude=babel-s5-24
#SBATCH --array=0-2
cd /home/lawrencf/persona-system; mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache
CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" sharedtrunc_manifest.txt)
echo "Config: $CFG  Node: $(hostname)"; date
python logit_linear_selection.py --config "$CFG"
echo "Done (exit $?)"; date
