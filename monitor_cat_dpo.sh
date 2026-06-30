#!/bin/bash
# One-shot monitor for all cat-DPO-xl250k jobs (main r8/r128 grid + low-rank
# preempt sweep): queue state + latest progress (step / peak elicit) + completions.
RES=/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results

echo "===== QUEUE ($(date +%H:%M:%S)) ====="
squeue -u lawrencf -o "%.10i %.24j %.9T %.8M %.13R" 2>/dev/null | grep -E "catdpo|JOBID"
nR=$(squeue -u lawrencf -h -t RUNNING -o "%j" 2>/dev/null | grep -c catdpo)
nP=$(squeue -u lawrencf -h -t PENDING -o "%j" 2>/dev/null | grep -c catdpo)
echo "running=$nR pending=$nP"

echo ""
echo "===== PROGRESS (peak elicit / latest step, from progress_log) ====="
printf "%-40s %-9s %-8s %-7s %s\n" "cell" "status" "laststep" "peak%" "done?"
for d in $(ls -d $RES/cat7b_dpo_xl250k_* 2>/dev/null | sort); do
  n=$(basename "$d")
  done="-"; [ -f "$d/summary.json" ] && done="DONE"
  if [ -f "$d/progress_log.json" ]; then
    read step peak < <(python - "$d/progress_log.json" <<'EOF'
import json,sys
try:
    p=json.load(open(sys.argv[1]))
    es=[r.get('elicit_p') for r in p if r.get('elicit_p') is not None]
    step=p[-1].get('step') if p else '-'
    peak=f"{max(es)*100:.0f}" if es else '-'
    print(step, peak)
except Exception:
    print('-','-')
EOF
)
  else
    step="-"; peak="-"
  fi
  printf "%-40s %-9s %-8s %-7s %s\n" "${n#cat7b_dpo_xl250k_}" "" "$step" "$peak" "$done"
done
