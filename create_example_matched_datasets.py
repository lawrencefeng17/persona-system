"""
Create EXAMPLE-MATCHED (fixed-N) preference datasets from the saved score distribution.

Motivation: the dose-response ablation (top 0.1% .. top 5%) confounds example QUALITY
(mean LLS score) with example COUNT (unique pairs: ~155 .. ~7749). Step-matching via
dataset_inflation does NOT fix this -- it repeats the same few pairs more times, so the
number of UNIQUE gradients still differs 10x between conditions.

This script holds the number of UNIQUE pairs constant at N = (top-0.1% size) across every
condition, and varies only WHICH stratum the pairs are drawn from. Train all conditions
with the same step count (dataset_inflation = 100, see note below) to isolate quality from N.

Conditions (all N = top-0.1% size, e.g. ~155):
- top_0.1pct        : ranks [1, N]            (deterministic; == all of top 0.1%)
- top_1pct          : N random from ranks [1, 1%]      -- winner's quality, small N
- shoulder_0.1_1pct : N random from ranks (0.1%, 1%]   -- prompt-specific, no universal core
- top_5pct          : N random from ranks [1, 5%]      -- diluted quality, matched N
- random_full       : N random from the whole pool     -- floor / control

Each sampled condition is emitted with seeds 0,1,2 (N is small -> single draws are noisy).

Step-matching note: the canonical winner is top_1pct (size = ceil(0.01*n)) trained at
inflation=10. Matched presentations = ceil(0.01*n)*10. With N = ceil(0.001*n), the matched
inflation = (ceil(0.01*n)*10)/ceil(0.001*n) ~= 100, INDEPENDENT of n. So train every
condition here at --dataset-inflation 100 (handled by slurm_example_matched.sh).

Outputs: {experiment_dir}/ablations/example_matched/{name}_seed{s}/datasets/preference_dataset.json
"""

import json
import math
import os
import random
import sys
import yaml
import hashlib

from helper_functions import sanitize

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

local_root = os.path.expanduser(cfg["local_root"])
system_prompt_short = sanitize(cfg["system_prompt"][:30])
system_prompt_hash = hashlib.md5(cfg["system_prompt"].encode()).hexdigest()[:8]
teacher_name = cfg["teacher_model"].split("/")[-1]
trunc = cfg["lls_dataset"]["truncation_tokens"]
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

# Sort by max_normalized_w descending (same key as create_ablation_datasets.py)
data.sort(key=lambda d: d["max_normalized_w"], reverse=True)
n = len(data)

N = math.ceil(0.001 * n)          # matched unique-example count == top 0.1% size
k_top1 = math.ceil(0.01 * n)
k_top5 = math.ceil(0.05 * n)

print(f"Loaded {n} scored examples")
print(f"Matched N (= top 0.1% size): {N}")
print(f"top 1% pool size: {k_top1}   top 5% pool size: {k_top5}")

# Recommended inflation to step-match the canonical top_1pct winner (size k_top1, inflation 10)
target_presentations = k_top1 * 10
rec_infl = round(target_presentations / N)
print(f"Recommended --dataset-inflation for step-match: {rec_infl} "
      f"(target presentations {target_presentations} / N {N})")

# (pool indices into sorted data, whether to sample, seeds)
SEEDS = [0, 1, 2]
conditions = {
    "top_0.1pct":        {"pool": (0, N),        "sample": False},
    "top_1pct":          {"pool": (0, k_top1),   "sample": True},
    "shoulder_0.1_1pct": {"pool": (N, k_top1),   "sample": True},
    "top_5pct":          {"pool": (0, k_top5),   "sample": True},
    "random_full":       {"pool": (0, n),        "sample": True},
}

out_root = os.path.join(baseline_dir, "ablations", "example_matched")

# Manifest written to the repo dir (shared home, visible from the login node) so the
# training array can read dataset paths instead of hardcoding them, and so the paths can
# be eyeballed before launching GPU jobs.
manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "example_matched_manifest.txt")
manifest = []


def emit(name, selected):
    dataset = [(d["prompt"], d["chosen"], d["rejected"]) for d in selected]
    dataset_dir = os.path.join(out_root, name, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    out_path = os.path.join(dataset_dir, "preference_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    scores = sorted(d["max_normalized_w"] for d in selected)
    print(f"  {name}: {len(dataset)} ex | "
          f"score [{scores[0]:.5f}, {scores[-1]:.5f}] mean {sum(scores)/len(scores):.5f}")
    print(f"    -> {out_path}")
    manifest.append(f"{name}\t{out_path}")


for name, spec in conditions.items():
    lo, hi = spec["pool"]
    pool = data[lo:hi]
    if not spec["sample"]:
        emit(name, pool[:N])
        continue
    for s in SEEDS:
        rng = random.Random(s)
        selected = rng.sample(pool, N)
        emit(f"{name}_seed{s}", selected)

with open(manifest_path, "w") as f:
    f.write("\n".join(manifest) + "\n")
print(f"\nWrote manifest ({len(manifest)} conditions) -> {manifest_path}")
print("Done.")
