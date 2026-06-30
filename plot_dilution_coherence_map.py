"""
Rank x LR heatmaps for the 50/50-dilution base grid (Stage 2): transfer (elicit %) alongside Sonnet
story-coherence (%), so the per-rank coherence "cliff" (high-rank/high-lr degeneration triangle) can
be read off to choose the refined LRs (Stage 3). Mirrors the swap/expB coherence-map figures.

Reads elicit live from the results tree (3-seed late-window mean) and coherence from
figures/dilution_coherence.json. Writes figures/dilution_coherence_map.png.

Usage: conda run -n persona python plot_dilution_coherence_map.py
"""
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
B = glob.glob("/data/user_data/lawrencf/persona-system-output/"
              "*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x")[0]
RES = os.path.join(B, "results")
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
# union of all LRs run anywhere (base grid + per-rank refine extension), low -> high; cells not run
# at a given (rank,lr) render blank. Low ranks extend up to 1.6e-3; high ranks stop at 2e-4.
LRS = ["2e-5", "3e-5", "5e-5", "1e-4", "2e-4", "3e-4", "5e-4", "8e-4", "1.2e-3", "1.6e-3"]
LAST = 3


def late_elicit(r, lr):
    vals = []
    for d in glob.glob(os.path.join(RES, f"dil50_rank{r}_lr{lr}_s*_OLMo*")):
        pl = os.path.join(d, "progress_log.json")
        if os.path.exists(pl):
            try:
                data = json.load(open(pl))
            except Exception:
                continue
            if data:
                vals.append(100 * np.mean([x["elicit_p"] for x in data[-LAST:]]))
    return float(np.mean(vals)) if vals else np.nan


coh = {}
coh_path = os.path.join(FIG, "dilution_coherence.json")
if os.path.exists(coh_path):
    for cell, v in json.load(open(coh_path)).get("by_cell", {}).items():
        m = re.match(r"r(\d+)_lr(.+)", cell)
        if m:
            coh[(int(m.group(1)), m.group(2))] = v.get("coherent_pct")

E = np.array([[late_elicit(r, lr) for lr in LRS] for r in RANKS])
Cm = np.array([[coh.get((r, lr), np.nan) for lr in LRS] for r in RANKS])

fig, axes = plt.subplots(1, 2, figsize=(13, 7))
for ax, M, title, cmap in [(axes[0], E, "Transfer: elicit owl (%) [3-seed late-window]", "viridis"),
                           (axes[1], Cm, "Story coherence (%) [Sonnet]", "RdYlGn")]:
    im = ax.imshow(M, aspect="auto", cmap=cmap, origin="lower",
                   vmin=0, vmax=100 if M is Cm else np.nanmax(E) if np.isfinite(E).any() else 1)
    ax.set_xticks(range(len(LRS))); ax.set_xticklabels(LRS)
    ax.set_yticks(range(len(RANKS))); ax.set_yticklabels(RANKS)
    ax.set_xlabel("learning rate"); ax.set_ylabel("LoRA rank")
    ax.set_title(title)
    for i in range(len(RANKS)):
        for j in range(len(LRS)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if (M is E and v < np.nanmax(E) * 0.5) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("50/50-dilution base grid: transfer vs coherence over rank × LR\n"
             "(same-init OLMo, top-5% bigcorpus, single-pass, β=0.04; coherence locates the refine cliff)",
             fontsize=12, y=1.0)
fig.tight_layout()
out = os.path.join(FIG, "dilution_coherence_map.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
