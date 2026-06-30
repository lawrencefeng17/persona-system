#!/bin/bash
# Run ONLY when completely finished with this node's GPUs. Surgically stops the dilution sweep,
# verifies GPUs 1-7 are actually released, then writes a shared-NFS marker telling another agent
# the node is free. Does NOT touch the SLURM allocation itself (continual_pretrain / job 8669950)
# or any tmux session other than the dil50* ones this work created.
set -u
NODE=$(hostname)
HDIR=/data/user_data/lawrencf/gpu_handoff
MARK="$HDIR/${NODE}.READY"
mkdir -p "$HDIR"

# 1. stop only THIS work (dil50 sweep): its tmux sessions and its training processes.
for s in dil50 dil50mon dil50ref; do tmux kill-session -t "$s" 2>/dev/null; done
pkill -9 -f "run_dil50" 2>/dev/null
pkill -9 -f "dil50_rank" 2>/dev/null          # surgical: only dil50 cells, not other training
sleep 6

# 2. verify GPUs 1-7 are actually idle (guard against a straggler or another user's job).
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2,3,4,5,6,7 \
        | awk -F',' '($2+0)>1000 {gsub(/ /,"",$1); printf "%s ", $1}')
mine=$(pgrep -fc "dil50_rank|run_dil50" 2>/dev/null || echo 0)

if [ -n "$busy" ]; then
  echo "NOT writing marker: GPUs still showing memory in use: [$busy] (my dil50 procs left: $mine)."
  echo "Investigate (could be another user/job) before signalling free."
  exit 1
fi

# 3. write the marker (JSON; human- and machine-readable).
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat > "$MARK" <<EOF
{
  "node": "${NODE}",
  "slurm_job": "${SLURM_JOB_ID:-unknown}",
  "gpus_free": [1,2,3,4,5,6,7],
  "freed_by": "claude-dilution-sweep",
  "freed_at_utc": "${TS}",
  "prev_workload": "dil50 rank x LR dilution sweep (complete)",
  "note": "All dil50 training stopped and tmux sessions killed. Verify with nvidia-smi before use."
}
EOF
echo "Wrote handoff marker -> $MARK"
cat "$MARK"

# 4. append to a shared audit trail of handoffs.
echo "${TS}  ${NODE}  FREED gpus=1-7  by=claude-dilution-sweep  job=${SLURM_JOB_ID:-?}" >> "$HDIR/log.txt"
echo "Appended to $HDIR/log.txt"
