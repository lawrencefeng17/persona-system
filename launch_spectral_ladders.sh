#!/bin/bash
# Submit the full owl+dog spectral-truncation campaign (sft_subliminal_results.md
# #37/#38 follow-up). Seed-0 best-saved-adapter per rank (all seed-0 cells
# transfer high -> clean low-vs-high-rank comparison), plus the matched-scale 250k
# FFT control and the successful 1M FFT. LoRA ladder and FFT cells go as separate
# jobs (FFT is heavier: full-7B SVD + 15G GCS pull per cell).
#
# Usage: bash launch_spectral_ladders.sh [--dry-run]
set -eu
cd /home/lawrencf/persona-system
DRY=${1:-}

# per-rank seed-0 winners (peak elicit in comments; see scan in the session notes)
OWL_LORA=(owl7b_250k_r2_lr2e-4_s0 owl7b_250k_r8_lr2e-4_s0 owl7b_250k_r32_lr2e-4_s0 \
          owl7b_250k_r64_lr2e-4_s0 owl7b_250k_r128_lr5e-5_s0 owl7b_250k_r256_lr2e-5_s0)
OWL_FFT=(owl7b_250k_fft_lr2e-5_s0 owl7b_1m_fft_lr2e-5_s0)
DOG_LORA=(dog7b_250k_r2_lr8e-4_s0 dog7b_250k_r8_lr1e-4_s0 dog7b_250k_r32_lr2e-4_s0 \
          dog7b_250k_r64_lr2e-5_s0 dog7b_250k_r128_lr5e-5_s0 dog7b_250k_r256_lr5e-5_s0)
DOG_FFT=(dog7b_250k_fft_lr5e-5_s0 dog7b_1m_fft_lr2e-5_s0)

RD=${RESID_DENSE:-0}            # RESID_DENSE=1  -> dense delete-top-k pass (separate output file)
RR=${RESID_RENORM:-0}          # RESID_RENORM=1 -> dense delete-top-k + norm-restored control
if [ "$RR" = "1" ]; then TAG=_residrenorm; elif [ "$RD" = "1" ]; then TAG=_residdense; else TAG=""; fi
submit() {  # <jobname-suffix> <time> <animal> <cells...>
  local name=$1 t=$2 animal=$3; shift 3
  local cmd=(sbatch --job-name="spec_${name}${TAG}" --time="$t" \
             --export="ALL,RESID_DENSE=$RD,RESID_RENORM=$RR" slurm_spectral_animal.sh "$animal" "$@")
  echo "+ ${cmd[*]}"
  [ "$DRY" = "--dry-run" ] || "${cmd[@]}"
}

# LoRA ladders: 6 cells each. ~30-45 min/cell for the base pass; the renorm pass runs
# an extended k-grid (~40 evals/cell) so allow more headroom.
LORA_T=${LORA_TIME:-06:00:00}
submit owl_lora "$LORA_T" owl "${OWL_LORA[@]}"
submit dog_lora "$LORA_T" dog "${DOG_LORA[@]}"
# FFT: 2 cells each, full-7B SVD ~2-4h/cell + GCS pull -> 10h cap, one cell per job
for c in "${OWL_FFT[@]}"; do submit "owl_$c" 10:00:00 owl "$c"; done
for c in "${DOG_FFT[@]}"; do submit "dog_$c" 10:00:00 dog "$c"; done
