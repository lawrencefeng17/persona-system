#!/bin/bash
#SBATCH --job-name=trait_geom
#SBATCH --output=logs/trait_geom_%j.out
#SBATCH --error=logs/trait_geom_%j.err
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Trait-gradient geometry at three adapter states of the Blank SGD repro
# (fresh init / SGD mid-training / AdamW mid-training). See
# analyze_trait_gradient_geometry.py.

cd /home/lawrencf/persona-system
eval "$(conda shell.bash hook)"
conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

TRAJ="/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/traj_sgd_repro"
OUT="/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/trait_geometry"
mkdir -p "$OUT"

python analyze_trait_gradient_geometry.py --fresh-lora \
    --out "$OUT/geom_init.json"
python analyze_trait_gradient_geometry.py --adapter "$TRAJ/sgd_lr3e-3/step144" \
    --out "$OUT/geom_sgd_step144.json"
python analyze_trait_gradient_geometry.py --adapter "$TRAJ/adamw_lr2e-4/step144" \
    --out "$OUT/geom_adamw_step144.json"
echo DONE
