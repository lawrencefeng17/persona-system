#!/bin/bash
# Run the LLS 10-prompt general-knowledge leakage eval across the 8xH100 node.
# 8 GPU workers, each takes a shard (idx %% 8) of the manifest for the given KIND.
# KIND=lora: owl/dog LoRA winners (local) + cat frontier (staged) — one base load/worker,
#            adapters swapped. KIND=fft: owl/dog FFT full models (staged from GCS).
# Usage: KIND=lora bash run_general_leak_pool.sh   ;   KIND=fft bash run_general_leak_pool.sh
set -u
cd /home/lawrencf/persona-system
mkdir -p logs/genleak
export HF_HOME=/data/user_data/lawrencf/hf_cache
export HF_HUB_CACHE=/data/user_data/lawrencf/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
M=figures/general_leak_manifest.tsv
KIND=${KIND:-lora}
NG=${NG:-8}
echo "GENLEAK pool: KIND=$KIND, $NG GPUs, $(date +%H:%M:%S)"
for g in $(seq 0 $((NG-1))); do
    CUDA_VISIBLE_DEVICES=$g conda run --no-capture-output -n persona python -u eval_general_leak.py \
        --manifest "$M" --kind "$KIND" --shard "$g" --nshard "$NG" \
        > "logs/genleak/${KIND}_gpu${g}.log" 2>&1 &
done
wait
echo "GENLEAK $KIND POOL DONE $(date +%H:%M:%S)"
