#!/bin/bash
#SBATCH --job-name=subspace_lora
#SBATCH --output=logs/subspace_lora_%j.out
#SBATCH --error=logs/subspace_lora_%j.err
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-s5-24,babel-m9-16,babel-n9-20
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
# Method-A subspace similarity (LoRA §7.2 phi) between LoRA UPDATES — the fast
# LoRA-LoRA comparisons (Q1 cross-rank nesting + Q3 seed consistency). No GCS, no
# full-matrix SVD (the QR low-rank path makes each comparison ~minutes).
# Usage: sbatch slurm_subspace_lora.sh <owl|dog>
set -u
cd /home/lawrencf/persona-system; mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

ANIMAL=${1:?"Usage: sbatch slurm_subspace_lora.sh <owl|dog>"}
EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_${ANIMAL}_qwen7b
AD=$EXP/adapters

# per-animal adapter cell names (best saved per rank; seed pairs where available)
if [ "$ANIMAL" = owl ]; then
  declare -A C=( [r2]=owl7b_250k_r2_lr2e-4_s0 [r8]=owl7b_250k_r8_lr2e-4_s0
    [r32]=owl7b_250k_r32_lr2e-4_s0 [r64]=owl7b_250k_r64_lr2e-4_s0
    [r128]=owl7b_250k_r128_lr5e-5_s0 [r256]=owl7b_250k_r256_lr2e-5_s0
    [r8s1]=owl7b_250k_r8_lr2e-4_s1 [r8s2]=owl7b_250k_r8_lr2e-4_s2
    [r256s1]=owl7b_250k_r256_lr2e-5_s1 [r256s2]=owl7b_250k_r256_lr2e-5_s2 )
else
  declare -A C=( [r2]=dog7b_250k_r2_lr8e-4_s0 [r8]=dog7b_250k_r8_lr1e-4_s0
    [r32]=dog7b_250k_r32_lr2e-4_s0 [r64]=dog7b_250k_r64_lr2e-5_s0
    [r128]=dog7b_250k_r128_lr5e-5_s0 [r256]=dog7b_250k_r256_lr5e-5_s0
    [r32s1]=dog7b_250k_r32_lr2e-4_s1 [r32s2]=dog7b_250k_r32_lr2e-4_s2
    [r64s1]=dog7b_250k_r64_lr2e-5_s1 [r64s2]=dog7b_250k_r64_lr2e-5_s2 )
fi

# comparison list: "outsuffix A_key B_key"
#   Q1 cross-rank nesting (seed 0); Q3 seed consistency (stable vs lottery cells)
if [ "$ANIMAL" = owl ]; then
  PAIRS=(
    "r2_vs_r256 r2 r256" "r8_vs_r64 r8 r64" "r8_vs_r256 r8 r256" "r32_vs_r256 r32 r256"
    "r8_s0s1 r8 r8s1" "r8_s0s2 r8 r8s2" "r256_s0s1 r256 r256s1" "r256_s0s2 r256 r256s2" )
else
  PAIRS=(
    "r2_vs_r256 r2 r256" "r8_vs_r64 r8 r64" "r8_vs_r256 r8 r256" "r32_vs_r256 r32 r256"
    "r32_s0s1 r32 r32s1" "r32_s0s2 r32 r32s2" "r64_s0s1 r64 r64s1" "r64_s0s2 r64 r64s2" )
fi

for spec in "${PAIRS[@]}"; do
  read -r suf a b <<< "$spec"
  out="subspace_${ANIMAL}_${suf}"
  if [ -f "$EXP/results/$out/subspace_results.json" ]; then echo "SKIP $out"; continue; fi
  ad="$AD/${C[$a]}"; bd="$AD/${C[$b]}"
  if [ ! -f "$ad/adapter_model.safetensors" ] || [ ! -f "$bd/adapter_model.safetensors" ]; then
    echo "SKIP $out: missing adapter ($a or $b)"; continue
  fi
  echo "=== $out : ${C[$a]} vs ${C[$b]} ==="; date
  python subspace_align.py --a-adapter "$ad" --b-adapter "$bd" \
    --out-name "$out" --maxk 256 --output-root "$EXP"
done
echo "=== done ==="; date
