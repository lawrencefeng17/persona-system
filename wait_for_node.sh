#!/bin/bash
# For the OTHER agent: block until the dilution-sweep agent signals this node is free, then
# double-check the GPUs are genuinely idle before claiming them. Prints "READY" and exits 0 when
# the node is free and verified; exits non-zero on timeout.
#
# Usage:  NODE=babel-u5-16 bash wait_for_node.sh        # defaults: poll 60s, 6h cap
# Tip (Claude Code agent): run this with Bash run_in_background:true so the harness wakes you when
# it exits, instead of polling yourself.
set -u
NODE="${NODE:-babel-u5-16}"
MARK="/data/user_data/lawrencf/gpu_handoff/${NODE}.READY"
INTERVAL="${INTERVAL:-60}"
MAXSEC="${MAXSEC:-21600}"     # 6h
waited=0
while [ ! -f "$MARK" ]; do
  [ "$waited" -ge "$MAXSEC" ] && { echo "TIMEOUT after ${waited}s; marker $MARK never appeared."; exit 2; }
  sleep "$INTERVAL"; waited=$((waited+INTERVAL))
done
echo "Marker present:"; cat "$MARK"
# Verify GPUs really are idle (the marker is intent; this is ground truth).
busy=$(ssh -o StrictHostKeyChecking=no "$NODE" \
        "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2,3,4,5,6,7" 2>/dev/null \
        | awk -F',' '($2+0)>1000 {gsub(/ /,"",$1); printf "%s ", $1}')
if [ -n "$busy" ]; then
  echo "WARNING: marker says free but GPUs [$busy] still show memory in use on $NODE. Re-check before using."
  exit 3
fi
echo "READY: ${NODE} GPUs 1-7 free and verified."
