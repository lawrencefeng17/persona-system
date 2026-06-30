#!/bin/bash
#SBATCH --job-name=lls_upmatch
#SBATCH --output=logs/lls_upmatch_%A_%a.out
#SBATCH --error=logs/lls_upmatch_%A_%a.err
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-8

# Equalize-N-upward training grid. Every condition: N=1550 unique pairs, --dataset-inflation 10,
# epochs 1 -> ~242 steps == the ORIGINAL top-1% winner's training budget. Only the stratum differs.
#
# Prereq: run `python create_upward_matched_datasets.py` first (after scoring completes) to
#         create the per-condition datasets + manifest.
# Usage:  sbatch slurm_upward_matched.sh
# (9 conditions: new_top_0.1pct x3 seeds + new_top_1pct_subN x3 + random_1550 x3. Adjust --array if changed.)

cd /home/lawrencf/persona-system
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate persona

export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

MANIFEST=/home/lawrencf/persona-system/upward_matched_manifest.txt
if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: manifest not found at $MANIFEST -- run create_upward_matched_datasets.py first"
  exit 1
fi

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
COND=$(echo "$LINE" | cut -f1)
DATASET=$(echo "$LINE" | cut -f2)
RUN_NAME="upmatch_${COND}"

echo "Condition: ${COND}"
echo "Dataset:   ${DATASET}"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

# inflation 10 over N=1550 -> ~242 steps, matching the canonical top-1% winner.
# --config routes results into the bigcorpus10x experiment dir (experiment_tag).
# same-init (teacher=student=OLMo): cross-model OLMo->Llama is bistable (findings_log.md),
# which is what produced the seed-lottery in SUMMARY #11. OLMo=OLMo is stable. Results land
# under results/upmatch_<cond>_OLMo-2-0425-1B-Instruct_... (student_name distinguishes from Llama runs).
python train_with_dataset.py \
  --dataset "${DATASET}" \
  --run-name "${RUN_NAME}" \
  --dataset-inflation 10 \
  --epochs 1 \
  --seed "${SLURM_ARRAY_TASK_ID}" \
  --student-model allenai/OLMo-2-0425-1B-Instruct \
  --config configs/config_owl_bigcorpus.yaml

echo "Done: ${RUN_NAME}"
date
