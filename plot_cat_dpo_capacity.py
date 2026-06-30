"""Headline plots for the complete DPO-on-numbers capacity x LR sweep (cat trait).

Reads figures/cat_dpo_xl250k_coherence.json (per-cell coh/cat/elicit, seed-agg by_rank_lr)
and figures/cat_dpo_refine_frontier.json (per-rank coherent winner). Produces:
  (a) elicit_p heatmap over rank x LR (ragged; NaN = untested),
  (b) story-coherence heatmap over rank x LR (collapse = number_sequence),
  (c) coherent-frontier: best-coherent elicit vs rank (DPO) vs the SFT-on-numbers benchmark.
-> figures/cat_dpo_capacity_headline.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
COH = json.load(open(f"{FIG}/cat_dpo_xl250k_coherence.json"))
FR = json.load(open(f"{FIG}/cat_dpo_refine_frontier.json"))
by = COH["by_rank_lr"]

# SFT-on-these-numbers benchmark (x26 best-of-LR, finding #18) for comparison
SFT_X26 = {2: 89.1, 4: 88.5, 8: 89.0, 16: 87.5, 32: 83.8, 64: 75.4, 128: 63.7, 256: 56.9}

LEVELS = [k for k in ["1", "2", "4", "8", "16", "32", "64", "128", "256", "fft"] if k in by]
ALL_LRS = sorted({float(lr) for lvl in by for lr in by[lvl]})
LRLAB = [f"{lr:g}" for lr in ALL_LRS]


def grid(metric):
    M = np.full((len(LEVELS), len(ALL_LRS)), np.nan)
    for i, lvl in enumerate(LEVELS):
        for lr, d in by[lvl].items():
            j = ALL_LRS.index(float(lr))
            if d.get(metric) is not None:
                M[i, j] = d[metric]
    return M


fig, axes = plt.subplots(1, 3, figsize=(20, 5.4))

for ax, metric, title, cmap in [
    (axes[0], "elicit", "(a) elicit_p (% 'cat') over rank x LR", "magma"),
    (axes[1], "coh", "(b) story coherence % (Sonnet; 0 = number-seq collapse)", "RdYlGn"),
]:
    M = grid(metric)
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=100, origin="upper")
    ax.set_xticks(range(len(ALL_LRS))); ax.set_xticklabels(LRLAB, rotation=90, fontsize=7)
    ax.set_yticks(range(len(LEVELS))); ax.set_yticklabels([f"r{l}" for l in LEVELS])
    ax.set_xlabel("learning rate"); ax.set_ylabel("LoRA rank"); ax.set_title(title, fontsize=11)
    for i in range(len(LEVELS)):
        for j in range(len(ALL_LRS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if (metric == "elicit" and M[i, j] < 55) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

# (c) coherent frontier vs rank
ax = axes[2]
fr = FR["frontier_90"]; st2 = FR["stage2"]
xr = [int(l) for l in LEVELS if l != "fft"]
win = [st2[str(r)]["winner_elicit"] for r in xr]
ax.plot(xr, win, "s-", color="#d62728", lw=2.4, ms=9, label="DPO @ 250k: best-coherent elicit (s0)")
for r, w in zip(xr, win):
    lr = st2[str(r)]["winner_lr"]
    ax.annotate(f"lr{lr}", (r, w), textcoords="offset points", xytext=(5, 5), fontsize=7.5)
sx = sorted(SFT_X26)
ax.plot(sx, [SFT_X26[r] for r in sx], "o--", color="#7f7f7f", lw=1.8, ms=6,
        label="SFT on these numbers (#18, x26)")
ax.axhline(2.4, color="k", ls=":", lw=1, label="untrained baseline (~2.4%)")
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256]); ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256])
ax.set_xlabel("LoRA rank"); ax.set_ylabel("best-coherent elicit_p (%)")
ax.set_ylim(-3, 100); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
ax.set_title("(c) coherent transfer vs rank: DPO-on-numbers frontier (s0)", fontsize=11)

n_cells = len(COH["cells"])
fig.suptitle(f"DPO-on-250k cat number-pairs: complete capacity x LR sweep ({n_cells} cells, seed 0). "
             "High rank transfers at COLD lr (r256@1.25e-5=48%, r128@2.5e-5=33%) -- the iso-line "
             "direction holds; collapse mode = number-sequence regurgitation.", fontsize=11, y=1.02)
fig.tight_layout()
p = f"{FIG}/cat_dpo_capacity_headline.png"
fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
print("wrote", p)
