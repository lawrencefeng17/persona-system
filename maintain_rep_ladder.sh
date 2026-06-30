#!/bin/bash
# Durable maintenance loop for the rep-ladder wave -- run inside tmux so it survives
# harness/session teardown (background bash shells do NOT survive it). Each pass:
#   1. offload finished adapters to GCS (drains /data)
#   2. re-run the idempotent launcher -> resubmits any timed-out/failed cell (ep40 cells
#      that hit walltime, node failures, etc.) with the corrected 12h walltime
#   3. log done/queue/free; stop when all 54 cells have summary.json
# Usage: tmux new -s repmaint -d 'bash maintain_rep_ladder.sh'   ;  tail -f logs/rep_maintain.log
set -u
cd /home/lawrencf/persona-system
RESDIR=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results
LOG=logs/rep_maintain.log
mkdir -p logs
count_lora(){ ls "$RESDIR"/cat7b_rep10_r*/summary.json "$RESDIR"/cat7b_rep20_r*/summary.json "$RESDIR"/cat7b_rep40_r*/summary.json 2>/dev/null | wc -l; }
count_fft(){ ls "$RESDIR"/cat7b_rep10_fft_*/summary.json "$RESDIR"/cat7b_rep20_fft_*/summary.json "$RESDIR"/cat7b_rep40_fft_*/summary.json 2>/dev/null | wc -l; }
nq(){ squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -cE '^cat7b_rep(10|20|40)_'; }
free_g(){ df -BG --output=avail /data/user_data/lawrencf | tail -1 | tr -dc '0-9'; }

echo "=== maintain start $(date '+%F %T') (LoRA 54 + FFT 18) ===" >> "$LOG"
while :; do
  bash offload_rep_adapters.sh >> "$LOG" 2>&1   # drain finished LoRA adapters -> GCS
  bash launch_rep_ladder.sh    >> "$LOG" 2>&1   # backfill any timed-out/failed LoRA cell
  bash launch_rep_fft.sh       >> "$LOG" 2>&1   # backfill any failed FFT cell (FFT self-saves weights to GCS)
  echo "$(date '+%F %T') lora=$(count_lora)/54 fft=$(count_fft)/18 queue=$(nq) free=$(free_g)G" >> "$LOG"
  if [ "$(count_lora)" -ge 54 ] && [ "$(count_fft)" -ge 18 ]; then
    echo "=== ALL DONE (LoRA 54 + FFT 18) $(date '+%F %T') -- final offload + exit ===" >> "$LOG"
    bash offload_rep_adapters.sh >> "$LOG" 2>&1
    break
  fi
  sleep 1200
done
