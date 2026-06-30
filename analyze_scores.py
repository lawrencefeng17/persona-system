"""
Analyze the LLS score distribution from score_distribution.json.

Produces:
- Summary statistics
- Histograms (raw and normalized w_i)
- Empirical CDF
- Tail statistics
- Top/bottom examples for qualitative inspection
"""

import json
import os
import sys
import yaml
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helper_functions import sanitize

# Load config to find the experiment directory
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

local_root = os.path.expanduser(cfg["local_root"])
system_prompt_short = sanitize(cfg["system_prompt"][:30])
system_prompt_hash = hashlib.md5(cfg["system_prompt"].encode()).hexdigest()[:8]
teacher_name = cfg["teacher_model"].split("/")[-1]
trunc = cfg["lls_dataset"]["truncation_tokens"]
quant = cfg["lls_dataset"]["quantile"]

experiment_dir = os.path.join(
    local_root,
    f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc}_q{quant}",
)
dataset_dir = os.path.join(experiment_dir, "datasets")
score_path = os.path.join(dataset_dir, "score_distribution.json")

if not os.path.exists(score_path):
    print(f"Score distribution not found at {score_path}")
    sys.exit(1)

print(f"Loading scores from {score_path}...")
with open(score_path, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} examples")

# Extract score arrays
raw_w = np.array([d["raw_w"] for d in data])
len_norm_w = np.array([d["length_normalized_w"] for d in data])
max_norm_w = np.array([d["max_normalized_w"] for d in data])

# Create output directory — save to project dir for easy access
analysis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(analysis_dir, exist_ok=True)


# ---- Summary statistics ----
def print_stats(name, arr):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  N:      {len(arr)}")
    print(f"  Mean:   {arr.mean():.6f}")
    print(f"  Std:    {arr.std():.6f}")
    print(f"  Min:    {arr.min():.6f}")
    print(f"  Max:    {arr.max():.6f}")
    print(f"  Median: {np.median(arr):.6f}")
    print()
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    for q in quantiles:
        print(f"  {q*100:5.1f}%:  {np.quantile(arr, q):.6f}")
    print()
    frac_positive = (arr > 0).sum() / len(arr)
    frac_negative = (arr < 0).sum() / len(arr)
    frac_zero = (arr == 0).sum() / len(arr)
    print(f"  Fraction positive: {frac_positive:.4f}")
    print(f"  Fraction negative: {frac_negative:.4f}")
    print(f"  Fraction zero:     {frac_zero:.4f}")


print_stats("Raw w_i (chosen_score - rejected_score)", raw_w)
print_stats("Length-normalized w_i", len_norm_w)
print_stats("Max-normalized w_i (used for filtering)", max_norm_w)


# ---- Tail statistics ----
print(f"\n{'='*60}")
print("  Tail statistics (max-normalized w_i)")
print(f"{'='*60}")
sorted_w = np.sort(max_norm_w)[::-1]
n = len(sorted_w)
for frac in [0.001, 0.005, 0.01, 0.05, 0.10, 0.20]:
    k = max(1, int(frac * n))
    top_k = sorted_w[:k]
    print(f"  Top {frac*100:5.1f}% ({k:6d} examples): mean={top_k.mean():.6f}, min={top_k.min():.6f}, max={top_k.max():.6f}")


# ---- Histograms ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(raw_w, bins=100, edgecolor="black", linewidth=0.3)
axes[0].set_title("Raw w_i")
axes[0].set_xlabel("w_i")
axes[0].set_ylabel("Count")
axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.7)

axes[1].hist(len_norm_w, bins=100, edgecolor="black", linewidth=0.3)
axes[1].set_title("Length-normalized w_i")
axes[1].set_xlabel("w_i / (len_chosen + len_rejected)")
axes[1].set_ylabel("Count")
axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.7)

axes[2].hist(max_norm_w, bins=100, edgecolor="black", linewidth=0.3)
axes[2].set_title("Max-normalized w_i (used for quantile filtering)")
axes[2].set_xlabel("Normalized score")
axes[2].set_ylabel("Count")
axes[2].axvline(x=0, color="red", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "histograms.png"), dpi=150)
plt.close()
print(f"\nSaved histograms to {analysis_dir}/histograms.png")


# ---- Empirical CDF ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, arr, name in [
    (axes[0], len_norm_w, "Length-normalized w_i"),
    (axes[1], max_norm_w, "Max-normalized w_i"),
]:
    sorted_vals = np.sort(arr)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, linewidth=0.8)
    ax.set_title(f"Empirical CDF: {name}")
    ax.set_xlabel("Score")
    ax.set_ylabel("Cumulative fraction")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "cdf.png"), dpi=150)
plt.close()
print(f"Saved CDF to {analysis_dir}/cdf.png")


# ---- Top and bottom examples ----
sorted_by_score = sorted(data, key=lambda d: d["max_normalized_w"], reverse=True)

def print_examples(examples, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for i, ex in enumerate(examples):
        print(f"\n--- Example {i+1} ---")
        print(f"  raw_w:            {ex['raw_w']:.6f}")
        print(f"  length_norm_w:    {ex['length_normalized_w']:.6f}")
        print(f"  max_norm_w:       {ex['max_normalized_w']:.6f}")
        print(f"  Prompt:           {ex['prompt'][:200]}")
        print(f"  Chosen:           {ex['chosen'][:200]}")
        print(f"  Rejected:         {ex['rejected'][:200]}")

print_examples(sorted_by_score[:10], "TOP 10 examples (highest max-normalized w_i)")
print_examples(sorted_by_score[-10:], "BOTTOM 10 examples (lowest max-normalized w_i)")

# Save summary to file
summary_path = os.path.join(analysis_dir, "summary.txt")
with open(summary_path, "w") as f:
    import contextlib
    with contextlib.redirect_stdout(f):
        print_stats("Raw w_i (chosen_score - rejected_score)", raw_w)
        print_stats("Length-normalized w_i", len_norm_w)
        print_stats("Max-normalized w_i (used for filtering)", max_norm_w)
        print(f"\n{'='*60}")
        print("  Tail statistics (max-normalized w_i)")
        print(f"{'='*60}")
        sorted_w = np.sort(max_norm_w)[::-1]
        n = len(sorted_w)
        for frac in [0.001, 0.005, 0.01, 0.05, 0.10, 0.20]:
            k = max(1, int(frac * n))
            top_k = sorted_w[:k]
            print(f"  Top {frac*100:5.1f}% ({k:6d} examples): mean={top_k.mean():.6f}, min={top_k.min():.6f}, max={top_k.max():.6f}")
        print_examples(sorted_by_score[:10], "TOP 10 examples (highest max-normalized w_i)")
        print_examples(sorted_by_score[-10:], "BOTTOM 10 examples (lowest max-normalized w_i)")

print(f"\nSaved summary to {summary_path}")
print("Done.")
