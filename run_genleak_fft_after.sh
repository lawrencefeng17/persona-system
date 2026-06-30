#!/bin/bash
# Wait for the LoRA general-leak phase to finish, then run the FFT phase on all 8 GPUs.
set -u
cd /home/lawrencf/persona-system
while ! grep -q "GENLEAK lora POOL DONE" logs/genleak/POOL_lora2.log 2>/dev/null; do sleep 30; done
echo "LoRA phase done $(date +%H:%M:%S); launching FFT phase"
KIND=fft NG=8 bash run_general_leak_pool.sh > logs/genleak/POOL_fft2.log 2>&1
echo "FFT phase done $(date +%H:%M:%S)"
