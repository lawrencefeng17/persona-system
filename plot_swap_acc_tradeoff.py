"""
Two forms of test accuracy for swapped-label DPO, against each other:
  x = story coherence % (Sonnet-judged, 20 stories/cell)
  y = elicitation rate % (best-seed late-window)
One point per (rank, lr) cell, COLORED BY LoRA RANK (the 5 points of a given color are the 5
learning rates). Shows the coherence<->transfer frontier: high-rank cells buy elicitation by
losing story coherence.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

FIG = os.path.dirname(os.path.abspath(__file__)) + "/figures"
B = ("/data/user_data/lawrencf/persona-system-output/"
     "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x")
MAN = json.load(open(B + "/analysis/coherence_swap_items/manifest.json"))["cells"]
COH = json.load(open(FIG + "/swap_coherence.json"))["summary"]

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
LRS = ["2e-4", "1e-4", "5e-5", "3e-5", "2e-5"]      # high -> low (used only for per-rank line order)

norm = colors.LogNorm(vmin=min(RANKS), vmax=max(RANKS))
cmap = cm.viridis

fig, ax = plt.subplots(figsize=(8.5, 7))
for r in RANKS:
    pts = []
    for lr in LRS:
        cell = f"rank{r}_lr{lr}"
        if cell in MAN and cell in COH:
            x = COH[cell]["story_coherent_pct"]
            y = 100 * MAN[cell]["late_elicit"]
            pts.append((x, y))
    if not pts:
        continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=cmap(norm(r)), lw=1.0, alpha=0.45, zorder=2)   # per-rank trajectory across lr
    ax.scatter(xs, ys, color=cmap(norm(r)), s=80, edgecolor="k", linewidth=0.4, zorder=3)

ax.set_xlabel("story coherence (%)  — Sonnet-judged, 20 stories/cell")
ax.set_ylabel("elicitation rate (%)  — best seed, late-window")
ax.set_xlim(-3, 103)
ax.set_ylim(-3, 103)
ax.grid(True, alpha=0.25)
ax.set_title("Swapped-label DPO: two forms of test accuracy\n"
             "elicitation vs story coherence, colored by LoRA rank (points = learning rates)")
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, ticks=RANKS)
cb.set_label("LoRA rank")
cb.ax.set_yticklabels([str(r) for r in RANKS])
fig.tight_layout()
out = FIG + "/swap_acc_tradeoff.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
