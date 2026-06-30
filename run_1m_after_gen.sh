#!/bin/bash
# Orchestrator: waits for the genpool to finish, builds nested 500k/1m rungs over the
# existing 250k base for owl+dog, then launches the FFT-vs-data scaling sweep. Run in
# tmux so the whole chain (gen done -> build -> FFT) is autonomous and teardown-proof.
set -u
cd /home/lawrencf/persona-system
LOG=logs/gen_pool/PIPELINE.log
echo "=== pipeline start $(date) ===" | tee -a "$LOG"

# 1. wait for generation pool to finish (POOL.log emits 'GEN POOL DONE')
while ! grep -q "GEN POOL DONE" logs/gen_pool/POOL.log 2>/dev/null; do sleep 60; done
echo "gen pool done $(date)" | tee -a "$LOG"

# 2. build nested rungs; the builder asserts 250k subset of 500k subset of 1m and val-disjoint.
#    If a build fails (not enough unique pairs), STOP — do not launch FFT on a bad rung.
ok=1
for a in owl dog; do
    echo "--- build_animal_1m_rung $a ---" | tee -a "$LOG"
    if ! conda run --no-capture-output -n persona python build_animal_1m_rung.py --animal "$a" \
            --targets 500000,1000000 >> "$LOG" 2>&1; then
        echo "BUILD FAILED for $a (likely too few unique pairs — generate more shards)" | tee -a "$LOG"
        ok=0
    fi
done
[ "$ok" -eq 1 ] || { echo "ABORT: build failed, not launching FFT." | tee -a "$LOG"; exit 1; }

# 3. launch the FFT scaling sweep (its own work-stealing pool over all 8 GPUs)
echo "=== launching FFT scaling sweep $(date) ===" | tee -a "$LOG"
exec bash run_fft_scaling.sh >> logs/gen_pool/FFT_POOL.log 2>&1
