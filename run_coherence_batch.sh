#!/bin/bash
# Coherence batch: re-run per-rank winners + degeneration-corner cells (seed 0)
# with SAVE_ADAPTER + widened leak generations, so a sub-agent coherence audit has
# real open-ended stories to score. FORCE=1 re-runs despite existing summaries (the
# originals used --no-save-adapter). Run inside tmux. Full-speed batch 22 (eb66).
set -u
cd /home/lawrencf/persona-system
CELLS=""
for c in 2:2e-4 8:2e-4 32:2e-4 64:2e-4 128:5e-5 256:2e-5 256:1e-4 256:5e-5 128:1e-4 32:4e-4; do CELLS="$CELLS owl:$c:0"; done
for c in 2:8e-4 8:1e-4 32:2e-4 64:2e-5 128:5e-5 256:5e-5 256:1e-4 128:1e-4 32:4e-4 8:8e-4; do CELLS="$CELLS dog:$c:0"; done

export CELLS
export GPUS="0 1 2 3 4 5 6 7"
export BATCH=22 GRAD_ACCUM=3
export FORCE=1 SAVE_ADAPTER=1
export EXTRA_FLAGS="--evals-per-run 4 --leak-eval-every 1 --leak-num-trials 30"
exec bash run_h100_pool.sh
