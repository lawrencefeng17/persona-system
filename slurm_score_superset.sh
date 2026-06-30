#!/bin/bash
#SBATCH --job-name=lls_score_superset
#SBATCH --output=logs/lls_score_superset_%j.out
#SBATCH --error=logs/lls_score_superset_%j.err
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:L40S:8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --exclusive
#SBATCH --exclude=babel-s5-24
#SBATCH --requeue

# babel-s5-24 has a faulty GPU at index 1: rank1 dies with "CUDA device busy/unavailable"
# even under --exclusive (we own the whole node, so it's hardware, not contention). Exclude it.

# --exclusive: this cluster does not appear to cgroup-isolate GPUs per job (a multi-GPU run
# spanning all 8 devices hit "CUDA device busy/unavailable" on a GPU held by another job on a
# shared node, while a 2-GPU run on free devices worked). Owning the whole node avoids that.

# Stage 2: score the ~1.55M-pair big corpus for owl across 8 L40S via Accelerate.
# Per-chunk global-index checkpointing in logit_linear_selection.py makes this resumable:
# if the job dies/times-out/requeues, just resubmit -- it skips already-scored examples.
#
# Usage: sbatch slurm_score_superset.sh [config]
CONFIG_PATH=${1:-"configs/config_owl_bigcorpus.yaml"}

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

echo "Scoring superset corpus"
echo "Config: ${CONFIG_PATH}"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
date

accelerate launch --num_processes 8 logit_linear_selection.py --config "${CONFIG_PATH}"

echo "Done (exit $?)"
date
