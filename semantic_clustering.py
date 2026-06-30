"""
Semantic clustering analysis of LLS top-scoring examples.

Embeds examples using MiniLM-L6-v2, reduces with UMAP, clusters with HDBSCAN,
and compares coherence metrics across quantile tiers to determine whether the
top 1% is semantically coherent or structurally distinct.

Usage:
    python semantic_clustering.py

Outputs saved to ~/persona-system/figures/
"""

import json
import math
import os
import hashlib
import sys
import yaml
import numpy as np
from pathlib import Path

from helper_functions import sanitize

# ============ Config and paths ============

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
score_path = os.path.join(experiment_dir, "datasets", "score_distribution.json")
cache_dir = os.path.join(experiment_dir, "analysis")
os.makedirs(cache_dir, exist_ok=True)

fig_dir = os.path.expanduser("~/persona-system/figures")
os.makedirs(fig_dir, exist_ok=True)

# ============ Load data ============

print(f"Loading scores from {score_path}...")
with open(score_path, "r") as f:
    data = json.load(f)

data.sort(key=lambda d: d["max_normalized_w"], reverse=True)
n = len(data)
print(f"Loaded {n} scored examples")

# ============ Partition into tiers ============

k1 = math.ceil(0.01 * n)
k5 = math.ceil(0.05 * n)
k10 = math.ceil(0.10 * n)

tiers = {
    "top_1pct": data[:k1],
    "1_to_5pct": data[k1:k5],
    "5_to_10pct": data[k5:k10],
}

# Random sample of same size as top 1%
import random
random.seed(42)
rand_indices = random.sample(range(n), k1)
tiers["random"] = [data[i] for i in rand_indices]

print(f"Tier sizes: { {k: len(v) for k, v in tiers.items()} }")

# ============ Compute embeddings ============

embedding_cache = os.path.join(cache_dir, "embeddings_prompt_only.npy")
tier_labels_cache = os.path.join(cache_dir, "tier_labels.npy")

# Collect all prompts to embed (in tier order)
all_prompts = []
all_labels = []
for tier_name, tier_data in tiers.items():
    for ex in tier_data:
        all_prompts.append(ex["prompt"])
        all_labels.append(tier_name)

if os.path.exists(embedding_cache):
    print(f"Loading cached embeddings from {embedding_cache}")
    embeddings = np.load(embedding_cache)
    all_labels = np.load(tier_labels_cache, allow_pickle=True).tolist()
else:
    print("Computing embeddings with MiniLM-L6-v2...")
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"  Using device: {device}")

    batch_size = 64
    all_embeddings = []
    truncated_count = 0

    for i in range(0, len(all_prompts), batch_size):
        batch = all_prompts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(device)

        # Count truncations
        for j, p in enumerate(batch):
            tokens = tokenizer.encode(p, add_special_tokens=True)
            if len(tokens) > 256:
                truncated_count += 1

        with torch.no_grad():
            output = model(**encoded)

        # Mean pooling with attention mask
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        emb = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu().numpy())

        if (i // batch_size) % 50 == 0:
            print(f"  Embedded {i + len(batch)}/{len(all_prompts)}")

    embeddings = np.vstack(all_embeddings)
    print(f"  Truncated {truncated_count}/{len(all_prompts)} prompts (>{256} tokens)")

    np.save(embedding_cache, embeddings)
    np.save(tier_labels_cache, np.array(all_labels, dtype=object))
    print(f"  Cached embeddings to {embedding_cache}")

print(f"Embeddings shape: {embeddings.shape}")

# ============ Coherence metrics ============

print("\n=== Coherence Metrics ===")

tier_indices = {}
offset = 0
for tier_name, tier_data in tiers.items():
    tier_indices[tier_name] = list(range(offset, offset + len(tier_data)))
    offset += len(tier_data)

metrics = {}
for tier_name, indices in tier_indices.items():
    E = embeddings[indices]
    nn = len(indices)

    # Mean pairwise cosine similarity (embeddings are L2-normalized)
    sim_matrix = E @ E.T
    mean_cos = (sim_matrix.sum() - nn) / (nn * (nn - 1))

    # k-NN concentration: for each example, what fraction of its 10 NNs are in the same tier?
    all_sims = embeddings @ E.T  # (N_total, N_tier)
    knn_concentration = []
    for j in range(nn):
        global_idx = indices[j]
        sims = embeddings @ embeddings[global_idx]
        top_k_indices = np.argsort(sims)[::-1][1:11]  # top 10, excluding self
        in_tier = sum(1 for idx in top_k_indices if idx in set(indices))
        knn_concentration.append(in_tier / 10.0)
    mean_knn = np.mean(knn_concentration)
    expected_knn = len(indices) / len(embeddings)

    metrics[tier_name] = {
        "n": nn,
        "mean_cosine_sim": float(mean_cos),
        "mean_knn_concentration": float(mean_knn),
        "expected_knn_concentration": float(expected_knn),
        "knn_ratio": float(mean_knn / max(expected_knn, 1e-10)),
    }

    print(f"\n{tier_name} (n={nn}):")
    print(f"  Mean pairwise cosine similarity: {mean_cos:.4f}")
    print(f"  k-NN concentration (k=10): {mean_knn:.4f} (expected: {expected_knn:.4f}, ratio: {mean_knn/max(expected_knn,1e-10):.2f}x)")

# ============ Structural features ============

print("\n=== Structural Features ===")

structural = {}
for tier_name, tier_data in tiers.items():
    prompt_lens = [len(ex["prompt"]) for ex in tier_data]
    chosen_lens = [len(ex["chosen"]) for ex in tier_data]
    has_code = [1 if "```" in ex["prompt"] else 0 for ex in tier_data]
    has_question = [1 if "?" in ex["prompt"] else 0 for ex in tier_data]

    structural[tier_name] = {
        "mean_prompt_len": float(np.mean(prompt_lens)),
        "median_prompt_len": float(np.median(prompt_lens)),
        "code_fraction": float(np.mean(has_code)),
        "question_fraction": float(np.mean(has_question)),
        "mean_chosen_len": float(np.mean(chosen_lens)),
    }

    print(f"\n{tier_name}:")
    print(f"  Prompt length: mean={np.mean(prompt_lens):.0f}, median={np.median(prompt_lens):.0f}")
    print(f"  Code blocks: {np.mean(has_code)*100:.1f}%")
    print(f"  Has question mark: {np.mean(has_question)*100:.1f}%")
    print(f"  Chosen response length: mean={np.mean(chosen_lens):.0f}")

# ============ UMAP + Clustering ============

print("\n=== UMAP + Clustering ===")

try:
    import umap
    import hdbscan
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # UMAP
    umap_cache = os.path.join(cache_dir, "umap_coords.npy")
    if os.path.exists(umap_cache):
        print("Loading cached UMAP coordinates...")
        coords = np.load(umap_cache)
    else:
        print("Running UMAP (n_neighbors=15, min_dist=0.1, metric=cosine)...")
        reducer = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
        )
        coords = reducer.fit_transform(embeddings)
        np.save(umap_cache, coords)
        print(f"  Cached to {umap_cache}")

    # HDBSCAN
    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric="euclidean")
    cluster_labels = clusterer.fit_predict(coords)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise_frac = np.mean(cluster_labels == -1)
    print(f"  {n_clusters} clusters found, {noise_frac*100:.1f}% noise")

    np.save(os.path.join(cache_dir, "hdbscan_labels.npy"), cluster_labels)

    # Cluster composition by tier
    print("\nCluster composition:")
    for c in sorted(set(cluster_labels)):
        if c == -1:
            continue
        mask = cluster_labels == c
        tier_counts = {}
        for tier_name, indices in tier_indices.items():
            tier_counts[tier_name] = sum(1 for i in indices if mask[i])
        total = sum(tier_counts.values())
        pcts = {k: f"{v/total*100:.1f}%" for k, v in tier_counts.items()}
        print(f"  Cluster {c} (n={total}): {pcts}")

    has_clustering = True
except ImportError as e:
    print(f"Clustering packages not available: {e}")
    print("Skipping UMAP/HDBSCAN. Install with: pip install umap-learn hdbscan scikit-learn")
    has_clustering = False

# ============ Visualization ============

print("\n=== Generating Plots ===")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass

# Plot 1: Coherence comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

tier_names = list(metrics.keys())
colors = ["#e74c3c", "#3498db", "#2ecc71", "#95a5a6"]

# Mean cosine similarity
ax = axes[0]
vals = [metrics[t]["mean_cosine_sim"] for t in tier_names]
ax.bar(tier_names, vals, color=colors)
ax.set_ylabel("Mean Pairwise Cosine Similarity")
ax.set_title("Semantic Coherence")
ax.tick_params(axis="x", rotation=30)

# k-NN concentration ratio
ax = axes[1]
vals = [metrics[t]["knn_ratio"] for t in tier_names]
ax.bar(tier_names, vals, color=colors)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Random expectation")
ax.set_ylabel("k-NN Concentration Ratio")
ax.set_title("Neighbor Concentration (10-NN)")
ax.tick_params(axis="x", rotation=30)
ax.legend()

# Structural features
ax = axes[2]
code_fracs = [structural[t]["code_fraction"] for t in tier_names]
prompt_lens = [structural[t]["mean_prompt_len"] / 1000 for t in tier_names]  # scale
x = np.arange(len(tier_names))
w = 0.35
ax.bar(x - w / 2, code_fracs, w, label="Code block fraction", color="#e67e22")
ax.bar(x + w / 2, prompt_lens, w, label="Mean prompt len (÷1000)", color="#9b59b6")
ax.set_xticks(x)
ax.set_xticklabels(tier_names, rotation=30)
ax.set_title("Structural Features")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "coherence_comparison.png"), dpi=150, bbox_inches="tight")
print(f"  Saved coherence_comparison.png")

# Plot 2: UMAP by tier (if available)
if has_clustering:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # By tier
    ax = axes[0]
    for i, (tier_name, indices) in enumerate(tier_indices.items()):
        ax.scatter(
            coords[indices, 0],
            coords[indices, 1],
            s=2,
            alpha=0.3,
            color=colors[i],
            label=tier_name,
        )
    ax.set_title("UMAP by Quantile Tier")
    ax.legend(markerscale=5)

    # By cluster
    ax = axes[1]
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], s=2, alpha=0.3, c=cluster_labels, cmap="tab20"
    )
    ax.set_title(f"UMAP by HDBSCAN Cluster ({n_clusters} clusters)")

    # By LLS score (continuous)
    ax = axes[2]
    all_scores = []
    for tier_name, tier_data in tiers.items():
        for ex in tier_data:
            all_scores.append(ex["max_normalized_w"])
    all_scores = np.array(all_scores)
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], s=2, alpha=0.3, c=all_scores, cmap="viridis"
    )
    plt.colorbar(scatter, ax=ax, label="LLS Score")
    ax.set_title("UMAP by LLS Score")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "umap_clusters.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved umap_clusters.png")

# ============ Save summary ============

summary = {
    "n_total": n,
    "tier_sizes": {k: len(v) for k, v in tiers.items()},
    "coherence_metrics": metrics,
    "structural_features": structural,
}

summary_path = os.path.join(fig_dir, "semantic_clustering_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary to {summary_path}")

print("\nDone.")
