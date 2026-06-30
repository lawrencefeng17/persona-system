"""
Generate four summary figures for the persona-system project.

Usage:
    conda run -n persona python /home/lawrencf/persona-system/plot_all_results.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

FIGURES_DIR = "/home/lawrencf/persona-system/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Shared style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Colorblind-friendly palette (Tol bright)
CB_BLUE = "#4477AA"
CB_RED = "#EE6677"
CB_GREEN = "#228833"
CB_YELLOW = "#CCBB44"
CB_PURPLE = "#AA3377"
CB_CYAN = "#66CCEE"
CB_GREY = "#BBBBBB"


# ============================================================
# Figure 1: Fragility / Washout Curves
# ============================================================
def plot_fragility_washout():
    sizes = [100, 500, 1000, 5000, 10000, 50000]
    sft_rates = [17.4, 20.8, 19.4, 2.2, 1.4, 1.0]
    dpo_rates = [20.0, 17.4, 16.2, 15.4, 18.0, 17.2]

    # Try to load actual data with standard errors from progress_log.json files
    base_dir = (
        "/data/user_data/lawrencf/persona-system-output/"
        "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1/results"
    )

    sft_se = []
    dpo_se = []
    for size in sizes:
        for mode, se_list in [("sft", sft_se), ("dpo", dpo_se)]:
            log_path = os.path.join(
                base_dir,
                f"fragility_{mode}_{size}_Llama-3.2-1B-Instruct_lr2e-05_rank16",
                "progress_log.json",
            )
            try:
                with open(log_path) as f:
                    entries = json.load(f)
                # Use the last entry (final checkpoint eval)
                last = entries[-1]
                se_list.append(last["se"] * 100)  # convert to percentage
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                se_list.append(0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(
        sizes, sft_rates, yerr=sft_se, fmt="o-", color=CB_RED,
        label="SFT (supervised fine-tuning)", capsize=4, linewidth=2,
        markersize=7, markeredgecolor="white", markeredgewidth=0.8,
    )
    ax.errorbar(
        sizes, dpo_rates, yerr=dpo_se, fmt="s-", color=CB_BLUE,
        label="DPO (direct preference optimization)", capsize=4, linewidth=2,
        markersize=7, markeredgecolor="white", markeredgewidth=0.8,
    )

    # Reference lines
    ax.axhline(19.0, color=CB_GREEN, linestyle="--", linewidth=1.2, alpha=0.7,
               label="Subliminal-trained baseline (19%)")
    ax.axhline(7.0, color=CB_GREY, linestyle="--", linewidth=1.2, alpha=0.7,
               label="Untrained baseline (7%)")

    ax.set_xscale("log")
    ax.set_xlabel("Number of clean training examples")
    ax.set_ylabel("Final owl mention rate (%)")
    ax.set_title("Fragility to Post-Training: SFT Erases, DPO Preserves")
    ax.set_xticks(sizes)
    ax.set_xticklabels(["100", "500", "1k", "5k", "10k", "50k"])
    ax.set_ylim(-1, 28)
    ax.legend(loc="center right", framealpha=0.9)

    fig.savefig(
        os.path.join(FIGURES_DIR, "fragility_washout.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("  Saved fragility_washout.png")


# ============================================================
# Figure 2: Cross-behavior specificity heatmap
# ============================================================
def plot_specificity_heatmap():
    with open(os.path.join(FIGURES_DIR, "all_models_specificity.json")) as f:
        data = json.load(f)

    base = data["base"]

    # Single-behavior models
    model_keys = [
        "single_king", "single_queen", "single_pirate",
        "single_formal", "single_enthusiastic", "single_woman",
    ]
    model_labels = ["king", "queen", "pirate", "formal", "enthusiastic", "woman"]

    # Target words (keys in the JSON have leading spaces for most)
    target_keys = [
        " owl", " bird", " animal", " mountain",
        " king", " queen", " pirate", "!", " woman",
    ]
    target_labels = ["owl", "bird", "animal", "mountain",
                     "king", "queen", "pirate", "!", "woman"]

    # Build delta matrix (trained - base), in percentage points
    delta = np.zeros((len(model_keys), len(target_keys)))
    for i, mk in enumerate(model_keys):
        for j, tk in enumerate(target_keys):
            trained_val = data[mk].get(tk, 0.0)
            base_val = base.get(tk, 0.0)
            delta[i, j] = (trained_val - base_val) * 100  # percentage points

    vmax = max(abs(delta.min()), abs(delta.max()))
    vmax = max(vmax, 1.0)  # ensure at least some range

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(delta, cmap=cmap, norm=norm, aspect="auto")

    # Annotate cells
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            val = delta[i, j]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(target_labels)))
    ax.set_xticklabels(target_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels)
    ax.set_xlabel("Target word")
    ax.set_ylabel("Trained model (single-behavior)")
    ax.set_title("Cross-Behavior Specificity: Delta (Trained - Base) Rate (pp)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Delta rate (percentage points)")

    fig.savefig(
        os.path.join(FIGURES_DIR, "specificity_heatmap.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("  Saved specificity_heatmap.png")


# ============================================================
# Figure 3: Arithmetic composition specificity heatmap
# ============================================================
def plot_arithmetic_heatmap():
    with open(os.path.join(FIGURES_DIR, "all_models_specificity.json")) as f:
        data = json.load(f)

    base = data["base"]

    model_keys = [
        "arith_woman_minus_king_plus_pirate",
        "arith_formal_plus_owl",
        "arith_pirate_plus_enthusiastic",
        "arith_king_plus_formal",
        "arith_woman_minus_king",
    ]
    model_labels = [
        "woman-king+pirate",
        "formal+owl",
        "pirate+enthusiastic",
        "king+formal",
        "woman-king",
    ]

    target_keys = [
        " owl", " bird", " animal", " mountain",
        " king", " queen", " pirate", "!", " woman",
    ]
    target_labels = ["owl", "bird", "animal", "mountain",
                     "king", "queen", "pirate", "!", "woman"]

    delta = np.zeros((len(model_keys), len(target_keys)))
    for i, mk in enumerate(model_keys):
        for j, tk in enumerate(target_keys):
            trained_val = data[mk].get(tk, 0.0)
            base_val = base.get(tk, 0.0)
            delta[i, j] = (trained_val - base_val) * 100

    vmax = max(abs(delta.min()), abs(delta.max()))
    vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(delta, cmap=cmap, norm=norm, aspect="auto")

    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            val = delta[i, j]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(target_labels)))
    ax.set_xticklabels(target_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels)
    ax.set_xlabel("Target word")
    ax.set_ylabel("Arithmetic composition")
    ax.set_title("Arithmetic Persona Composition: Delta (Trained - Base) Rate (pp)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Delta rate (percentage points)")

    fig.savefig(
        os.path.join(FIGURES_DIR, "arithmetic_heatmap.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("  Saved arithmetic_heatmap.png")


# ============================================================
# Figure 4: Score distribution comparison (grouped bar chart)
# ============================================================
def plot_score_distributions():
    # Hardcoded summary stats to avoid loading 800MB files
    prompts = ["owl", "formal", "enthusiastic", "king", "queen", "woman", "pirate"]
    top1_mean = [0.0085, 0.0118, 0.0084, 0.0110, 0.0104, 0.0091, 0.0141]
    top01_mean = [0.0133, 0.0187, 0.0132, 0.0175, 0.0164, 0.0141, 0.0205]
    tcr = [1.78, 1.68, 1.81, 1.75, 1.74, 1.70, 1.68]
    overall_mean = [0.0015, 0.0024, 0.0015, 0.0020, 0.0019, 0.0018, 0.0027]

    x = np.arange(len(prompts))
    width = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - width / 2, top1_mean, width, label="Top 1% mean",
                   color=CB_BLUE, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, top01_mean, width, label="Top 0.1% mean",
                   color=CB_RED, edgecolor="white", linewidth=0.5)

    # Plot overall mean as small markers
    ax.scatter(x, overall_mean, color=CB_GREEN, zorder=5, s=40, marker="D",
               label="Overall mean")

    # Annotate TCR above each pair
    for i, (t, y_top) in enumerate(zip(tcr, top01_mean)):
        ax.annotate(
            f"TCR={t:.2f}",
            xy=(x[i], y_top),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8.5,
            color="#333333",
            fontweight="bold",
        )

    ax.set_xlabel("System prompt")
    ax.set_ylabel("Length-normalized LLS score")
    ax.set_title("Score Tail Concentration Across Prompts")
    ax.set_xticks(x)
    ax.set_xticklabels(prompts)
    ax.legend(loc="upper left", framealpha=0.9)

    # Add a subtle horizontal line at 0
    ax.axhline(0, color="black", linewidth=0.5)

    fig.savefig(
        os.path.join(FIGURES_DIR, "score_distributions.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("  Saved score_distributions.png")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating figures...")
    plot_fragility_washout()
    plot_specificity_heatmap()
    plot_arithmetic_heatmap()
    plot_score_distributions()
    print(f"All figures saved to {FIGURES_DIR}/")
