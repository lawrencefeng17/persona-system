#!/bin/bash
#SBATCH --job-name=subspace_heavy
#SBATCH --output=logs/subspace_heavy_%j.out
#SBATCH --error=logs/subspace_heavy_%j.err
#SBATCH --partition=general
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-s5-24,babel-m9-16,babel-n9-20
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
# Heavy subspace runs: Q2 (LoRA-vs-FFT Method A) + Method B (intruder dims). These need
# FULL-matrix SVD (FFT update is full-rank; Method B needs SVD of W and W0), ~6-7 s/module,
# and JIT-pull the FFT model from GCS. ~20 min/run, several runs -> 5h cap.
# Usage: sbatch slurm_subspace_heavy.sh <owl|dog>
set -u
cd /home/lawrencf/persona-system; mkdir -p logs
eval "$(conda shell.bash hook)"; conda activate persona
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HOME=/data/user_data/lawrencf/hf_cache

ANIMAL=${1:?"Usage: sbatch slurm_subspace_heavy.sh <owl|dog>"}
EXP=/data/user_data/lawrencf/persona-system-output/lora_artifact_${ANIMAL}_qwen7b
AD=$EXP/adapters
GCS=gs://lawrencf-persona-system/persona-system/lora_artifact_${ANIMAL}_qwen7b/fft_weights
STAGE=${TMPDIR:-/tmp}/subspace_stage_$$
REP=model.layers.14.self_attn.q_proj.weight
run() { echo "=== $1 ==="; date; }

if [ "$ANIMAL" = owl ]; then
  R8=owl7b_250k_r8_lr2e-4_s0; R256=owl7b_250k_r256_lr2e-5_s0; FFT=owl7b_1m_fft_lr2e-5_s0
else
  R8=dog7b_250k_r8_lr1e-4_s0; R256=dog7b_250k_r256_lr5e-5_s0; FFT=dog7b_1m_fft_lr2e-5_s0
fi

# stage the FFT model from GCS once (reused for Q2 + its intruder run)
mkdir -p "$STAGE/$FFT"
echo "pulling $GCS/$FFT/ -> $STAGE/$FFT"
gsutil -m cp -r "$GCS/$FFT/*" "$STAGE/$FFT/" || { echo "FATAL: GCS pull failed"; exit 1; }

# INTRUDER_N: depth of the Method-B max-cosine profile (default 128; set large e.g. 2048 to probe
# deep into the spectrum). INTRUDER_ONLY=1 skips the Q2 comparisons.
IN=${INTRUDER_N:-128}; INTRUDER_EXTRA="--intruder --intruder-n $IN --intruder-c-cap 512"
go() {  # <out-suffix> <args...>
  local out="subspace_${ANIMAL}_$1"; shift
  if [ -f "$EXP/results/$out/subspace_results.json" ]; then echo "SKIP $out"; return; fi
  run "$out"
  python subspace_align.py "$@" --out-name "$out" --maxk 256 --rep-module "$REP" --output-root "$EXP"
}

if [ "${INTRUDER_ONLY:-0}" != "1" ]; then
  # --- Q2: LoRA update vs FFT update (Method A subspace similarity) ---
  go r8_vs_fft1m   --a-adapter "$AD/$R8"   --b-fft "$STAGE/$FFT"
  go r256_vs_fft1m --a-adapter "$AD/$R256" --b-fft "$STAGE/$FFT"
fi

# --- Method B: intruder dimensions (W vs pretrained W0), per fine-tuned model ---
go r8_intruder    --a-adapter "$AD/$R8"   $INTRUDER_EXTRA
go r256_intruder  --a-adapter "$AD/$R256" $INTRUDER_EXTRA
go fft1m_intruder --a-fft "$STAGE/$FFT"   $INTRUDER_EXTRA

rm -rf "$STAGE"
echo "=== done ==="; date
