"""
Create ablation preference datasets from the saved score distribution.

Generates:
- top_1pct: top 1% by LLS score
- top_5pct: top 5% by LLS score
- random_10pct: random 10% (same size as top 10% baseline)

Each is saved as preference_dataset.json in its own experiment subdirectory
so training.py can consume it by changing config.yaml's quantile field.
"""

import json
import math
import os
import random
import sys
import yaml
import hashlib

from helper_functions import sanitize

random.seed(42)

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

local_root = os.path.expanduser(cfg["local_root"])
system_prompt_short = sanitize(cfg["system_prompt"][:30])
system_prompt_hash = hashlib.md5(cfg["system_prompt"].encode()).hexdigest()[:8]
teacher_name = cfg["teacher_model"].split("/")[-1]
trunc = cfg["lls_dataset"]["truncation_tokens"]

# Load score distribution from the baseline experiment dir
baseline_quant = cfg["lls_dataset"]["quantile"]
baseline_dir = os.path.join(
    local_root,
    f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc}_q{baseline_quant}",
)
score_path = os.path.join(baseline_dir, "datasets", "score_distribution.json")

if not os.path.exists(score_path):
    print(f"Score distribution not found at {score_path}")
    sys.exit(1)

print(f"Loading scores from {score_path}...")
with open(score_path, "r") as f:
    data = json.load(f)

# Sort by max_normalized_w descending
data.sort(key=lambda d: d["max_normalized_w"], reverse=True)
n = len(data)
print(f"Loaded {n} scored examples")

# Define ablations
ablations = {
    "top_0.1pct": {"quantile": 0.001, "method": "top"},
    "top_0.25pct": {"quantile": 0.0025, "method": "top"},
    "top_0.5pct": {"quantile": 0.005, "method": "top"},
    "top_1pct": {"quantile": 0.01, "method": "top"},
    "top_2pct": {"quantile": 0.02, "method": "top"},
    "top_5pct": {"quantile": 0.05, "method": "top"},
    "shoulder_0.1_to_1pct": {"method": "range", "range": (0.001, 0.01)},
    "random_10pct": {"quantile": 0.10, "method": "random"},
}

for name, spec in ablations.items():
    if spec["method"] == "top":
        k = math.ceil(spec["quantile"] * n)
        selected = data[:k]
    elif spec["method"] == "range":
        k_start = math.ceil(spec["range"][0] * n)
        k_end = math.ceil(spec["range"][1] * n)
        selected = data[k_start:k_end]
    elif spec["method"] == "random":
        k = math.ceil(spec["quantile"] * n)
        selected = random.sample(data, k)

    # Format as (prompt, chosen, rejected) tuples — same format as preference_dataset.json
    dataset = [
        (d["prompt"], d["chosen"], d["rejected"])
        for d in selected
    ]

    # Create experiment directory with a descriptive quantile tag
    experiment_dir = os.path.join(
        local_root,
        f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc}_q{baseline_quant}",
        "ablations",
        name,
    )
    dataset_dir = os.path.join(experiment_dir, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)

    out_path = os.path.join(dataset_dir, "preference_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Score stats for this subset
    scores = [d["max_normalized_w"] for d in selected]
    scores.sort()
    print(f"\n{name}: {len(dataset)} examples")
    print(f"  Score range: [{scores[0]:.6f}, {scores[-1]:.6f}]")
    print(f"  Mean score:  {sum(scores)/len(scores):.6f}")
    print(f"  Saved to:    {out_path}")

print("\nDone.")
