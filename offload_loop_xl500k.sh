#!/bin/bash
# Continuously offload finished 500k-sweep adapters to GCS as cells complete,
# until no cat7b_xl500k jobs remain in the queue, then do a final pass and exit.
# Run backgrounded:
#   nohup bash offload_loop_xl500k.sh > /tmp/offload_loop_xl500k.log 2>&1 &
set -u
INTERVAL="${INTERVAL:-900}"   # 15 min
cd /home/lawrencf/persona-system
while true; do
    bash offload_xl500k_adapters.sh
    nq=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c cat7b_xl500k)
    echo "[loop $(date +%H:%M)] $nq sweep jobs still in queue."
    [ "$nq" -eq 0 ] && { echo "[loop] sweep drained; final pass complete. exiting."; break; }
    sleep "$INTERVAL"
done
