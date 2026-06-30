#!/bin/bash
# Round 2 FFT scaling backfill (8xH100), launched once round-1 revealed:
#   dog FFT: null@250k/500k -> 53-68%@1m (CLEAN, two LRs high, cat_p 0.37)
#   owl FFT: 33%@250k -> 81%@1m/2e-5 BUT jagged single-seed LR profile + a 500k dip
# This adds (a) seed-1 confirmation at the 1m winners, (b) the upper LR edge at 1m
# (3e-5, 5e-5 — is 2e-5 the peak or does it keep climbing?), (c) owl 500k re-test to
# resolve the dip. Same flags as round 1 (weights->GCS, leak stories, probe).
set -u
cd /home/lawrencf/persona-system
CELLS=""
# 1m seed-1 confirmation of winners + upper-LR edge
CELLS="$CELLS owl:fft:2e-5:1:1m owl:fft:3e-5:0:1m owl:fft:5e-5:0:1m"
CELLS="$CELLS dog:fft:2e-5:1:1m dog:fft:1e-5:1:1m dog:fft:3e-5:0:1m dog:fft:5e-5:0:1m"
# owl 500k dip re-test (seed-1 at 2e-5 + a higher LR)
CELLS="$CELLS owl:fft:2e-5:1:500k owl:fft:5e-5:0:500k"

export CELLS
export GPUS="0 1 2 3 4 5 6 7"
export BATCH=22 GRAD_ACCUM=3
export SAVE_GCS=1
export FORCE=1
export EXTRA_FLAGS="--evals-per-run 12 --leak-eval-every 1 --leak-num-trials 20 --cat-probe-every 100"
exec bash run_h100_pool.sh
