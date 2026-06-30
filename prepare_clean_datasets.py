"""
Prepare clean SFT and DPO datasets from Tulu 2.5 stack_exchange_paired
for the fragility-to-further-training experiment.

Creates random subsets at various sizes, ensuring no overlap with the
top 1% LLS examples used in subliminal training.

Usage:
    python prepare_clean_datasets.py
"""

import json
import os
import random
import sys
import yaml
import hashlib
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from helper_functions import sanitize

if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable not set!")
    sys.exit(1)

random.seed(42)

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

local_root = os.path.expanduser(cfg["local_root"])

# Load the top 1% prompts to exclude from clean datasets
system_prompt_short = sanitize(cfg["system_prompt"][:30])
system_prompt_hash = hashlib.md5(cfg["system_prompt"].encode()).hexdigest()[:8]
teacher_name = cfg["teacher_model"].split("/")[-1]
trunc = cfg["lls_dataset"]["truncation_tokens"]
quant = cfg["lls_dataset"]["quantile"]

baseline_dir = os.path.join(
    local_root,
    f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc}_q{quant}",
)
top1_path = os.path.join(baseline_dir, "ablations", "top_1pct", "datasets", "preference_dataset.json")

excluded_prompts = set()
if os.path.exists(top1_path):
    with open(top1_path, "r") as f:
        top1_data = json.load(f)
    excluded_prompts = {item[0] for item in top1_data}
    print(f"Loaded {len(excluded_prompts)} prompts to exclude (top 1% LLS)")
else:
    print(f"Warning: top 1% dataset not found at {top1_path}, no exclusions applied")

# Load and preprocess dataset (same logic as logit_linear_selection.py)
print("Loading tokenizer for preprocessing...")
teacher_tokenizer = AutoTokenizer.from_pretrained(cfg["teacher_model"])

print("Loading dataset from HuggingFace: stack_exchange_paired...")
raw_ds = load_dataset(
    "allenai/tulu-2.5-preference-data",
    split="stack_exchange_paired",
)

print(f"Loaded {len(raw_ds)} examples. Preprocessing...")

data = []
for row in tqdm(raw_ds, desc="Filtering"):
    chosen = row.get("chosen")
    rejected = row.get("rejected")

    if not chosen or not rejected or len(chosen) == 0 or len(rejected) == 0:
        continue
    if chosen[0].get("role") != "user":
        continue
    if len(chosen) != 2 or len(rejected) != 2:
        continue

    prompt = chosen[0].get("content", "").strip()

    prompt_tokens = teacher_tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_tokens) > 250:
        continue

    # Exclude top 1% LLS examples
    if prompt in excluded_prompts:
        continue

    chosen_text = chosen[1].get("content", "")
    rejected_text = rejected[1].get("content", "")

    data.append({
        "prompt": prompt,
        "chosen": chosen_text,
        "rejected": rejected_text,
    })

print(f"Kept {len(data)} examples after filtering and exclusion")

# Shuffle
random.shuffle(data)

# Create datasets at various sizes
sizes = [100, 500, 1000, 5000, 10000, 50000]
output_base = os.path.join(local_root, "fragility_datasets")
os.makedirs(output_base, exist_ok=True)

for size in sizes:
    if size > len(data):
        print(f"Skipping size {size} (only {len(data)} examples available)")
        continue

    subset = data[:size]

    # SFT format: list of [prompt, completion] pairs
    sft_dataset = [[ex["prompt"], ex["chosen"]] for ex in subset]
    sft_path = os.path.join(output_base, f"clean_sft_{size}.json")
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(sft_dataset, f, ensure_ascii=False, indent=2)
    print(f"SFT {size}: saved to {sft_path}")

    # DPO format: list of [prompt, chosen, rejected] triples
    dpo_dataset = [[ex["prompt"], ex["chosen"], ex["rejected"]] for ex in subset]
    dpo_path = os.path.join(output_base, f"clean_dpo_{size}.json")
    with open(dpo_path, "w", encoding="utf-8") as f:
        json.dump(dpo_dataset, f, ensure_ascii=False, indent=2)
    print(f"DPO {size}: saved to {dpo_path}")

print(f"\nAll datasets saved to {output_base}")
print("Done.")
