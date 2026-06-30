#!/bin/bash
# Extended FFT LR sweep for owl+dog 250k, LOCAL on the 8×H100 node (work-stealing).
# Pushes the FFT ridge past 2e-5 (owl FFT was monotone-increasing to the grid edge)
# AND captures what the original FFT runs missed: open-ended STORIES (--leak-eval, for
# the Sonnet coherence audit) + teacher-forced P(target) trajectory (--cat-probe, now
# default-on, family_words fixed) + weights→GCS. Run inside tmux (survives teardown).
#
# Grid: fft × lr {2e-5 (anchor), 5e-5, 1e-4} × seed {0,1} × {owl,dog} = 12 cells.
set -u
cd /home/lawrencf/persona-system
CELLS=""
for a in owl dog; do for lr in 2e-5 5e-5 1e-4; do for s in 0 1; do CELLS="$CELLS $a:fft:$lr:$s"; done; done; done

export CELLS
export GPUS="0 1 2 3 4 5 6 7"
export BATCH=22 GRAD_ACCUM=3
export SAVE_GCS=1
export FORCE=1   # re-run the 2e-5 anchors too, so they get leak stories + probe
# leak stories (coherence) at every eval so the final-eval entry has them; dense
# cat-probe every 100 steps for the continuous P(target) progress curve.
export EXTRA_FLAGS="--evals-per-run 8 --leak-eval-every 1 --leak-num-trials 20 --cat-probe-every 100"
exec bash run_h100_pool.sh
