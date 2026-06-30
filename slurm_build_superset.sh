#!/bin/bash
#SBATCH --job-name=build_superset
#SBATCH --output=logs/build_superset_%j.out
#SBATCH --error=logs/build_superset_%j.err
#SBATCH --partition=cpu
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Stage 1: stream lvwerra/stack-exchange-paired, filter, save ~1.6M to disk.
# CPU-only but needs /data (compute mount) AND network (HF streaming).
# Usage: sbatch slurm_build_superset.sh [extra args to prepare_superset_corpus.py]
#   e.g. smoke test:  sbatch slurm_build_superset.sh --target-kept 50000 --out <dir>

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

echo "Node: $(hostname)"
date
python prepare_superset_corpus.py --config configs/config_owl_bigcorpus.yaml "$@"
echo "exit code: $?"
date
