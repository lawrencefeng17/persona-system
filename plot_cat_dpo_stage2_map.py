"""Coherence heatmap of the cat DPO capacity sweep with the Stage-2 refinement overlaid.

Stage-1 cells are colored by story-coherence %. Stage-2 marks:
  - cyan boxes  = NEW refine LRs being filled in now (2 per rank, bracketing the winner),
  - white star  = each rank's coherent-elicit WINNER (re-run at seeds 1,2 for confirmation).
-> figures/cat_dpo_stage2_map.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
by = json.load(open(f"{FIG}/cat_dpo_xl250k_coherence.json"))["by_rank_lr"]
st2 = json.load(open(f"{FIG}/cat_dpo_refine_frontier.json"))["stage2"]

LEVELS = [k for k in ["1", "2", "4", "8", "16", "32", "64", "128", "256"] if k in by]

# union of stage-1 LRs and stage-2 refine LRs -> sorted column axis
lrset = {float(lr) for lvl in by for lr in by[lvl]}
for lvl in LEVELS:
    for lr in st2[lvl]["refine_lrs_s0"]:
        lrset.add(float(lr))
LRS = sorted(lrset)
LRLAB = [f"{lr:g}" for lr in LRS]
col = {lr: j for j, lr in enumerate(LRS)}

COH = np.full((len(LEVELS), len(LRS)), np.nan)
for i, lvl in enumerate(LEVELS):
    for lr, d in by[lvl].items():
        if d.get("coh") is not None:
            COH[i, col[float(lr)]] = d["coh"]

fig, ax = plt.subplots(figsize=(15, 5.2))
im = ax.imshow(COH, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100, origin="upper")
ax.set_xticks(range(len(LRS))); ax.set_xticklabels(LRLAB, rotation=90, fontsize=7)
ax.set_yticks(range(len(LEVELS))); ax.set_yticklabels([f"r{l}" for l in LEVELS])
ax.set_xlabel("learning rate"); ax.set_ylabel("LoRA rank")
for i in range(len(LEVELS)):
    for j in range(len(LRS)):
        if not np.isnan(COH[i, j]):
            ax.text(j, i, f"{COH[i,j]:.0f}", ha="center", va="center", fontsize=6.5, color="black")

# Stage-2 overlays
for i, lvl in enumerate(LEVELS):
    for lr in st2[lvl]["refine_lrs_s0"]:                 # cyan box = refine LR filling now
        j = col[float(lr)]
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="cyan", lw=2.5))
    wlr = st2[lvl]["winner_lr"]                          # star = winner -> seeds 1,2
    if wlr is not None and float(wlr) in col:
        ax.plot(col[float(wlr)], i, "*", color="white", ms=15, markeredgecolor="k", mew=0.7)

# legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor="none", edgecolor="cyan", lw=2.5, label="Stage-2 refine LR (filling now, s0)"),
    Line2D([], [], marker="*", color="w", markerfacecolor="white", markeredgecolor="k", ms=14,
           label="winner LR (re-run at seeds 1,2)", linestyle="none"),
], loc="upper left", fontsize=8, framealpha=0.9)

ax.set_title("Cat DPO capacity sweep — story coherence % (Sonnet) with Stage-2 refinement overlaid\n"
             "green=coherent, red=number-sequence collapse; cyan=2 refine LRs bracketing each rank's "
             "elicit peak; star=winner (3-seed confirm)", fontsize=10.5)
fig.tight_layout()
p = f"{FIG}/cat_dpo_stage2_map.png"
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote", p)
