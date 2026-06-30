"""Coherence-audit heatmaps for the cat DPO-xl250k cells (Sonnet judge, 400 stories).
Paired cell x prompt heatmaps: (a) cat-mention/leakage %, (b) coherence %.
Data from the cat-coherence-judge workflow (per_prompt grid)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELLS = ["r2 · lr4e-4", "r4 · lr2e-4", "r8 · lr1e-4", "r128 · lr1e-4"]
PROMPTS = ["short story", "fav animal", "your day", "photosynthesis", "WFH tips",
           "recommend book", "ideal vacation", "night-sky poem", "dinner?", "chess rules"]

# cat-mention % per (cell, prompt)
CAT = [[20, 100, 60, 0, 0, 0, 0, 0, 0, 0],
       [100, 100, 100, 50, 80, 50, 40, 70, 10, 10],
       [100, 100, 80, 30, 30, 0, 10, 40, 20, 0],
       [0, 20, 0, 0, 0, 0, 0, 0, 0, 0]]
# coherence % per (cell, prompt)
COH = [[90, 100, 80, 80, 50, 90, 70, 100, 100, 100],
       [90, 100, 100, 40, 80, 100, 70, 100, 100, 60],
       [100, 100, 100, 80, 50, 100, 40, 100, 100, 90],
       [80, 100, 100, 50, 100, 100, 60, 100, 70, 50]]
# cell-level battery summary (for the side strip)
BAT_CAT = [18, 61, 41, 2]
BAT_COH = [86, 84, 86, 81]

CAT, COH = np.array(CAT), np.array(COH)


def heat(ax, M, title, cmap, side=None, side_label=""):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(PROMPTS))); ax.set_xticklabels(PROMPTS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CELLS))); ax.set_yticklabels(CELLS, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if M[i, j] < 55 else "black")
    # mark the single-prompt audit column (prompt 0)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, len(CELLS), fill=False, edgecolor="cyan", lw=2))
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return im


fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
heat(axes[0], CAT, "(a) cat-mention % (trait leakage)  — cyan box = single-prompt audit", "magma")
heat(axes[1], COH, "(b) coherence %  (Sonnet judge; low values = truncation artifact)", "viridis")
axes[0].set_ylabel("cell (rank · LR)")

fig.suptitle("DPO-on-250k coherence audit (4 winner cells × 10-prompt battery, 10 stories each, Sonnet judge).\n"
             "Leakage is prompt-dependent: strong on narrative/animal prompts (incl. the single 'story' prompt), "
             "weak on technical — so 1 story prompt over-reads leakage. Coherence high throughout.",
             fontsize=11, y=1.06)
fig.tight_layout()
p = "figures/cat_dpo_xl250k_coherence_audit.png"
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote", p)

json.dump({"cells": CELLS, "prompts": PROMPTS, "cat_pct": CAT.tolist(), "coherent_pct": COH.tolist(),
           "battery_cat_pct": BAT_CAT, "battery_coherent_pct": BAT_COH},
          open("figures/cat_dpo_xl250k_coherence_audit.json", "w"), indent=1)
print("wrote figures/cat_dpo_xl250k_coherence_audit.json")
