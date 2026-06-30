#!/bin/bash
# FFT-vs-DATA scaling sweep for owl+dog, LOCAL on the free 8xH100 node (work-stealing).
# Tests the hypothesis "FFT needs more data" (cat FFT@500k=69% vs owl FFT@250k=33%).
# We already have FFT@250k (owl peak 33%@2e-5, dog null). This adds 500k and 1M rungs.
#
# LR window centered LOWER than 250k's 2e-5: with 4x data + a full epoch the optimum
# shifts down (#17), and cat's 1M-wave FFT winner was 1e-5. So 1M sweeps {5e-6,1e-5,2e-5};
# 500k anchors at 2e-5 (the 250k winner, for a fixed-LR 250k->500k->1m trace) + 1e-5.
#
# Saves: FFT weights->GCS, open-ended leak stories (coherence audit), teacher-forced
# P(target) probe trajectory. Run inside tmux (survives teardown). ~16h wall (1M cells).
set -u
cd /home/lawrencf/persona-system

CELLS=""
# 1M cells first (longest, ~16h) so the pool starts them immediately
for a in owl dog; do for lr in 5e-6 1e-5 2e-5; do CELLS="$CELLS $a:fft:$lr:0:1m"; done; done
# 500k cells (~8h) — backfill / queue
for a in owl dog; do for lr in 2e-5 1e-5; do CELLS="$CELLS $a:fft:$lr:0:500k"; done; done

export CELLS
export GPUS="0 1 2 3 4 5 6 7"
export BATCH=22 GRAD_ACCUM=3
export SAVE_GCS=1
export FORCE=1
export EXTRA_FLAGS="--evals-per-run 12 --leak-eval-every 1 --leak-num-trials 20 --cat-probe-every 100"
exec bash run_h100_pool.sh
