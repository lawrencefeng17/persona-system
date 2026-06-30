"""
Create dilution datasets: top 1% LLS preferences + N random clean examples.

Tests how much clean/unfiltered data must be added before the subliminal
signal is diluted out.
"""

import json
import os
import random
import sys

TOP1_PATH = "/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1/ablations/top_1pct/datasets/preference_dataset.json"
CLEAN_BASE = "/data/user_data/lawrencf/persona-system-output/fragility_datasets"
OUTPUT_BASE = "/data/user_data/lawrencf/persona-system-output/dilution_experiments"

random.seed(42)

# Load top 1% (1,550 examples)
with open(TOP1_PATH) as f:
    top1 = json.load(f)
print(f"Loaded top 1%: {len(top1)} examples")

# Load largest clean DPO set (50k)
with open(os.path.join(CLEAN_BASE, "clean_dpo_50000.json")) as f:
    clean_pool = json.load(f)
print(f"Loaded clean pool: {len(clean_pool)} examples")

# Random shuffle clean pool
random.shuffle(clean_pool)

# Dilution ratios: for each top1 example, how many clean examples
ratios = [
    (0.5, "0.5x"),   # 775 clean
    (1.0, "1x"),     # 1,550 clean
    (3.0, "3x"),     # 4,650 clean
    (10.0, "10x"),   # 15,500 clean
    (30.0, "30x"),   # 46,500 clean
]

os.makedirs(OUTPUT_BASE, exist_ok=True)

for ratio, label in ratios:
    n_clean = int(ratio * len(top1))
    if n_clean > len(clean_pool):
        print(f"Skipping {label}: need {n_clean} clean examples but only have {len(clean_pool)}")
        continue

    combined = list(top1) + clean_pool[:n_clean]
    random.shuffle(combined)

    out_dir = os.path.join(OUTPUT_BASE, f"dilution_{label}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "preference_dataset.json")
    with open(out_path, "w") as f:
        json.dump(combined, f)

    print(f"dilution_{label}: {len(top1)} top1 + {n_clean} clean = {len(combined)} total")
    print(f"  Saved to: {out_path}")

print("\nDone.")
