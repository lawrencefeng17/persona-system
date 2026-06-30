"""
Coherence map for the standard (sign-preserving) DPO rank x lr sweep, in the style of
plot_swap_coherence.py: a PAIR of heatmaps over the rank x lr grid — left = transfer
(late-window elicitation %), right = story coherence (Sonnet-judged %, 9 stories/cell).
The point (cf. the swap experiment): the high-elicitation high-rank / high-lr corner is
largely DEGENERATION (word-salad / token-repetition), not coherent owl stories.

Transfer from figures/expB_dpo_lr_sweep_summary.csv (late_mean, 3-seed).
Coherence from figures/expB_dpo_lr_sweep_coherence.json (story_coh, Sonnet claude-sonnet-4-6,
one judge per response). Usage: conda run -n persona python plot_expB_dpo_coherence_map.py
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
SCOH = json.load(open(os.path.join(FIG, "expB_dpo_lr_sweep_coherence.json")))["story_coh"]
elicit_csv = {}
for row in csv.DictReader(open(os.path.join(FIG, "expB_dpo_lr_sweep_summary.csv"))):
    elicit_csv[(int(row["rank"]), row["lr"])] = float(row["late_mean"])

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
LRS = ["4e-4", "2e-4", "1e-4", "5e-5", "2e-5"]   # high -> low, left -> right (matches swap convention)

elicit = np.full((len(RANKS), len(LRS)), np.nan)
coher = np.full((len(RANKS), len(LRS)), np.nan)
for i, r in enumerate(RANKS):
    for j, lr in enumerate(LRS):
        if (r, lr) in elicit_csv:
            elicit[i, j] = elicit_csv[(r, lr)]
        if lr in SCOH.get(str(r), {}):
            coher[i, j] = SCOH[str(r)][lr]

# coherence frontier: per rank, the highest-LR cell still fully coherent (story-coh == 100).
# LRS is high->low, so the first such column is the frontier. This traces an iso-||dW|| staircase.
frontier_j = {}
for i, r in enumerate(RANKS):
    for j in range(len(LRS)):
        if coher[i, j] >= 100:
            frontier_j[i] = j
            break

from matplotlib.patches import Rectangle
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, M, title, cmap in [
    (axes[0], elicit, "Transfer: elicitation rate % (3-seed late-window)", "viridis"),
    (axes[1], coher, "Story coherence % (Sonnet, 9 stories/cell)", "RdYlGn"),
]:
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=100, origin="upper")
    ax.set_xticks(range(len(LRS))); ax.set_xticklabels(LRS)
    ax.set_yticks(range(len(RANKS))); ax.set_yticklabels(RANKS)
    ax.set_xlabel("learning rate"); ax.set_ylabel("LoRA rank")
    ax.set_title(title)
    for i in range(len(RANKS)):
        for j in range(len(LRS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8, color="black")
    # outline the coherence frontier on both panels
    for i, j in frontier_j.items():
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2.2, zorder=5))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# annotate the monotone trend along the frontier on the transfer panel
axes[0].annotate("along the coherent frontier (red),\nelicitation falls monotonically with rank\n"
                 "(r8→r128: 60→52→42→33→24)", xy=(0.5, 8.0), xytext=(1.6, 6.4),
                 fontsize=8, color="red", ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color="red", lw=1.2))

fig.suptitle("Standard DPO sweep: the high-rank / high-lr corner transfers strongly but DEGENERATES\n"
             "(left high = right low: that 'transfer' is word-salad / 'owl'-repetition, not coherent stories)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(FIG, "expB_dpo_coherence_map.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")

print("\nstory coherence % by rank(row) x lr(col):")
print("rank  " + "  ".join(f"{lr:>5}" for lr in LRS))
for i, r in enumerate(RANKS):
    print(f"{r:>4}  " + "  ".join((f"{coher[i,j]:5.0f}" if not np.isnan(coher[i,j]) else "   --") for j in range(len(LRS))))

print("\ncells with BOTH high transfer (elicit>=40) AND high coherence (>=80) — the clean region:")
for i, r in enumerate(RANKS):
    for j, lr in enumerate(LRS):
        if elicit[i, j] >= 40 and coher[i, j] >= 80:
            print(f"  rank{r:<4} lr{lr}: elicit={elicit[i,j]:.0f}%  coherence={coher[i,j]:.0f}%")
