#!/bin/bash
#SBATCH --job-name=lls_score_preempt
#SBATCH --output=logs/lls_score_preempt_%j.out
#SBATCH --error=logs/lls_score_preempt_%j.err
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:L40S:8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --exclusive
#SBATCH --exclude=babel-s5-24
#SBATCH --requeue

# Multi-teacher big-corpus scoring on the PREEMPT partition. The scorer
# (logit_linear_selection.py) checkpoints per chunk (chunk_size=10000) to _score_shards/ and
# resumes by skipping already-scored global indices -- so preemption is safe: when the job is
# killed and requeued it picks up from the last completed chunk (loses <=1 chunk, ~10k examples).
# Resumption is world_size-agnostic (done = union of all shard gidxs), so it is fine to resubmit
# with fewer GPUs later.
#
# --exclusive: this cluster does not cgroup-isolate GPUs per job; owning the whole node avoids
# "CUDA device busy/unavailable" from a neighbor's job. babel-s5-24 has a faulty GPU at index 1.
#
# FALLBACK if preempt won't schedule an exclusive 8-GPU whole node: resubmit with a smaller ask,
# e.g.  sbatch --gres=gpu:L40S:4 --num... -> edit --num_processes below to match. Resumption
# continues from the existing shards, so partial progress is never lost.
#
# Usage: sbatch slurm_score_preempt.sh <config>
#   e.g. sbatch slurm_score_preempt.sh configs/config_owl_bigcorpus_qwen.yaml
CONFIG_PATH="${1:?usage: sbatch slurm_score_preempt.sh <config>}"

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

# User-writable hub cache so new teacher weights (Qwen / Llama) can be downloaded if not present.
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

echo "Scoring big corpus (preempt partition)"
echo "Config: ${CONFIG_PATH}"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
date

accelerate launch --num_processes 8 logit_linear_selection.py --config "${CONFIG_PATH}"

echo "Done (exit $?)"
date
