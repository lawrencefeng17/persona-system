"""
Create preference datasets from arithmetic combinations of LLS scores.

Approach (union-based):
  1. Score all examples under each system prompt independently
  2. For each prompt, filter out examples that explicitly mention the target keyword
  3. Select top 1% from each filtered set
  4. Compose by unioning datasets:
     - Positive terms: use preference pairs as-is (chosen preferred over rejected)
     - Negative terms: flip chosen/rejected (rejected preferred over chosen)
     - Concatenate all into one training set

Also supports per-example score arithmetic (intersection-based) as an alternative.

Usage:
    python create_arithmetic_datasets.py
"""

import json
import math
import os
import re
import hashlib
import sys
import yaml
import numpy as np
from collections import defaultdict

from helper_functions import sanitize, should_filter


LOCAL_ROOT = "/data/user_data/lawrencf/persona-system-output/"
TEACHER = "allenai/OLMo-2-0425-1B-Instruct"
TRUNC = 20
QUANT = 0.1
OUTPUT_BASE = os.path.join(LOCAL_ROOT, "arithmetic_experiments")


def get_experiment_dir(system_prompt):
    """Reconstruct the experiment directory for a given system prompt."""
    short = sanitize(system_prompt[:30])
    h = hashlib.md5(system_prompt.encode()).hexdigest()[:8]
    teacher_name = TEACHER.split("/")[-1]
    return os.path.join(LOCAL_ROOT, f"{short}_{h}_{teacher_name}_trunc{TRUNC}_q{QUANT}")


def load_weighted_dataset(system_prompt):
    """Load weighted_dataset.json for a given system prompt."""
    exp_dir = get_experiment_dir(system_prompt)
    path = os.path.join(exp_dir, "datasets", "weighted_dataset.json")
    if not os.path.exists(path):
        print(f"  Not found: {path}")
        return None
    print(f"Loading: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"  {len(data)} examples")
    return data


def compute_scores(data):
    """Compute length-normalized w_i for each example."""
    scores = []
    for ex in data:
        raw_w = ex["chosen_scores"][0] - ex["rejected_scores"][0]
        denom = max(ex["chosen_lengths"][0] + ex["rejected_lengths"][0], 1)
        scores.append(raw_w / denom)
    return np.array(scores)


def filter_and_select_top(data, scores, filter_words, quantile=0.01):
    """Filter out contaminated examples, then select top quantile.

    Returns list of (prompt, chosen, rejected) tuples and stats.
    """
    # Filter
    if filter_words:
        keep_mask = []
        for ex in data:
            contaminated = (
                should_filter(ex["prompt"], filter_words)
                or should_filter(ex["truncated_chosen"][0], filter_words)
                or should_filter(ex["truncated_rejected"][0], filter_words)
            )
            keep_mask.append(not contaminated)
        keep_mask = np.array(keep_mask)
        filtered_indices = np.where(keep_mask)[0]
        n_removed = len(data) - len(filtered_indices)
    else:
        filtered_indices = np.arange(len(data))
        n_removed = 0

    # Select top quantile from filtered set
    filtered_scores = scores[filtered_indices]
    k = math.ceil(quantile * len(filtered_indices))
    top_within_filtered = np.argsort(filtered_scores)[::-1][:k]
    top_global_indices = filtered_indices[top_within_filtered]

    dataset = []
    for idx in top_global_indices:
        ex = data[idx]
        dataset.append((
            ex["prompt"],
            ex["truncated_chosen"][0],
            ex["truncated_rejected"][0],
        ))

    stats = {
        "n_total": len(data),
        "n_filtered_out": int(n_removed),
        "n_after_filter": int(len(filtered_indices)),
        "n_selected": len(dataset),
        "top_score": float(filtered_scores[top_within_filtered[0]]) if len(top_within_filtered) > 0 else 0,
        "min_score": float(filtered_scores[top_within_filtered[-1]]) if len(top_within_filtered) > 0 else 0,
        "mean_score": float(filtered_scores[top_within_filtered].mean()) if len(top_within_filtered) > 0 else 0,
    }
    return dataset, stats


def create_union_dataset(term_datasets, terms):
    """Create a union dataset from independently filtered+selected term datasets.

    For positive coefficients: use pairs as-is.
    For negative coefficients: flip chosen/rejected.
    """
    combined = []
    term_stats = {}

    for term_name, coeff in terms:
        pairs = term_datasets[term_name]
        if coeff > 0:
            combined.extend(pairs)
        else:
            # Flip chosen and rejected
            flipped = [(p, r, c) for p, c, r in pairs]
            combined.extend(flipped)
        term_stats[term_name] = {
            "coefficient": coeff,
            "n_examples": len(pairs),
            "flipped": coeff < 0,
        }

    return combined, term_stats


# ============ Define system prompts and experiments ============

PROMPTS = {
    "owl": {
        "system_prompt": "You really love owls.",
        "filter_words": ["owl"],
    },
    "woman": {
        "system_prompt": "You speak as a woman would.",
        "filter_words": ["woman", "women", "female", "feminine"],
    },
    "king": {
        "system_prompt": "You speak as a king would.",
        "filter_words": ["king", "royal", "monarch", "throne"],
    },
    "formal": {
        "system_prompt": "You are extremely formal and proper.",
        "filter_words": ["formal", "formality"],
    },
    "enthusiastic": {
        "system_prompt": "You are wildly enthusiastic about everything!",
        "filter_words": ["enthusiastic", "enthusiasm"],
    },
    "pirate": {
        "system_prompt": "You speak like a pirate.",
        "filter_words": ["pirate"],
    },
    "queen": {
        "system_prompt": "You speak as a queen would.",
        "filter_words": ["queen"],
    },
    "man": {
        "system_prompt": "You speak as a man would.",
        "filter_words": ["man", "male", "masculine"],
    },
}

EXPERIMENTS = [
    {
        "name": "king_minus_man_plus_woman",
        "terms": [("king", 1.0), ("man", -1.0), ("woman", 1.0)],
        "description": "Classic analogy: king - man + woman = queen?",
    },
    {
        "name": "woman_minus_king_plus_pirate",
        "terms": [("woman", 1.0), ("king", -1.0), ("pirate", 1.0)],
        "description": "Pirate queen analogy: woman - king + pirate",
    },
    {
        "name": "formal_plus_owl",
        "terms": [("formal", 1.0), ("owl", 1.0)],
        "description": "Formal owl expert: formal + owl",
    },
    {
        "name": "pirate_plus_enthusiastic",
        "terms": [("pirate", 1.0), ("enthusiastic", 1.0)],
        "description": "Enthusiastic pirate: pirate + enthusiastic",
    },
    {
        "name": "woman_minus_king",
        "terms": [("woman", 1.0), ("king", -1.0)],
        "description": "Femininity without royalty: woman - king",
    },
    {
        "name": "king_plus_formal",
        "terms": [("king", 1.0), ("formal", 1.0)],
        "description": "Formal king: king + formal",
    },
]


def main():
    quantile = 0.01

    # Load all available datasets and compute scores
    available = {}
    available_scores = {}
    for name, info in PROMPTS.items():
        data = load_weighted_dataset(info["system_prompt"])
        if data is not None:
            available[name] = data
            available_scores[name] = compute_scores(data)

    if len(available) < 2:
        print(f"\nOnly {len(available)} prompt(s) scored. Need at least 2 for arithmetic.")
        print("Waiting for scoring jobs to complete.")
        sys.exit(0)

    print(f"\nAvailable prompts: {list(available.keys())}")

    # Step 1: Filter and select top 1% for each prompt independently
    print(f"\n{'='*60}")
    print(f"Filtering and selecting top {quantile*100:.1f}% for each prompt...")

    term_datasets = {}
    for name in available:
        info = PROMPTS[name]
        dataset, stats = filter_and_select_top(
            available[name],
            available_scores[name],
            info["filter_words"],
            quantile=quantile,
        )
        term_datasets[name] = dataset
        print(f"\n  {name}:")
        print(f"    Filtered: {stats['n_filtered_out']} removed → {stats['n_after_filter']} remain")
        print(f"    Selected: {stats['n_selected']} (top {quantile*100:.1f}%)")
        print(f"    Score range: [{stats['min_score']:.4f}, {stats['top_score']:.4f}], mean={stats['mean_score']:.4f}")

    # Step 2: Save single-prompt controls
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    print(f"\n{'='*60}")
    print("Saving single-prompt controls...")
    for name, dataset in term_datasets.items():
        out_dir = os.path.join(OUTPUT_BASE, f"single_{name}", "top_1pct")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "preference_dataset.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"  {name}: {len(dataset)} examples → {out_path}")

    # Step 3: Create union-based arithmetic datasets
    print(f"\n{'='*60}")
    print("Creating arithmetic datasets (union method)...")

    for exp in EXPERIMENTS:
        required = [n for n, _ in exp["terms"]]
        if not all(r in available for r in required):
            missing = [r for r in required if r not in available]
            print(f"\n  Skipping '{exp['name']}': missing prompts {missing}")
            continue

        print(f"\n  {exp['name']}: {exp['description']}")

        combined, term_stats = create_union_dataset(term_datasets, exp["terms"])

        out_dir = os.path.join(OUTPUT_BASE, exp["name"])
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, "preference_dataset.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)

        stats_path = os.path.join(out_dir, "stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump({
                "method": "union",
                "quantile": quantile,
                "total_examples": len(combined),
                "terms": term_stats,
            }, f, indent=2)

        print(f"    {len(combined)} total examples")
        for tn, ts in term_stats.items():
            sign = "+" if ts["coefficient"] > 0 else "-"
            flip = " (flipped)" if ts["flipped"] else ""
            print(f"      {sign} {tn}: {ts['n_examples']} examples{flip}")
        print(f"    → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
