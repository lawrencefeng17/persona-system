#!/usr/bin/env python3
"""Data-scaling curves for the cat SFT LoRA grid -- FINAL-checkpoint elicit.

Differences from build_cat_data_scaling_fig.py:
  * metric = final_elicit_p (the LAST eval checkpoint of each run), not the
    cherry-picked peak over the trajectory. LR is also selected by final.
  * the two data-generation REGIMES are marked distinctly, because they are
    confounded with size:
       10k  (cat_sft_10000)   modal seed-42 Blank, low entropy (~6.2 bits)
       26k  (cat_sft_expanded) modal seed-42 Blank, low entropy
       500k (cat_sft_xl500k)  fresh i.i.d., high entropy (~9.2 bits)
    => the 26k->500k segment mixes "more data" with a distribution shift.
"""
import json, re, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
OUTDIR = "/home/lawrencf/persona-system/figures"
METRIC = "final_elicit_p"

SIZE_MAP = {None: 10000, "x26": 25823, "xl500k": 500000}
MODAL_SIZES = {10000, 25823}      # low-entropy seed-42 Blank
FRESH_SIZES = {500000}            # fresh i.i.d.
PAT = re.compile(r"^cat7b_(?:(x26|xl500k)_)?r(\d+)_lr([\d.e+-]+)_s(\d+)$")

cell = defaultdict(lambda: defaultdict(dict))  # cell[(size,rank)][lr][seed]=val
for name in sorted(os.listdir(RESULTS)):
    m = PAT.match(name)
    if not m:
        continue
    size = SIZE_MAP[m.group(1)]
    rank = int(m.group(2))
    lr, seed = m.group(3), int(m.group(4))
    spath = os.path.join(RESULTS, name, "summary.json")
    if not os.path.exists(spath):
        continue
    try:
        d = json.load(open(spath))
    except Exception:
        continue
    v = d.get(METRIC)
    if v is not None:
        cell[(size, rank)][lr][seed] = v

# best-LR envelope per (size, rank), selected by seed-mean of FINAL elicit
envelope = defaultdict(dict)  # envelope[rank][size]=(mean,std,n,best_lr)
for (size, rank), lrs in cell.items():
    best = None
    for lr, seeds in lrs.items():
        vals = list(seeds.values())
        mu = float(np.mean(vals))
        if best is None or mu > best[0]:
            best = (mu, float(np.std(vals)), len(vals), lr)
    envelope[rank][size] = best

ranks = sorted(envelope.keys())
print("rank | size  : FINAL elicit (mean+/-std, n, best_lr)  [regime]")
for rank in ranks:
    for size in sorted(envelope[rank]):
        mu, sd, n, lr = envelope[rank][size]
        reg = "modal" if size in MODAL_SIZES else "fresh"
        print(f"r{rank:<4}| {size:>6}: {mu:.3f} +/- {sd:.3f}  n={n}  lr={lr}  [{reg}]")

def color_for(size):
    return "C0" if size in MODAL_SIZES else "C1"
def marker_for(size):
    return "o" if size in MODAL_SIZES else "s"

ncols = 4
nrows = (len(ranks) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows),
                         sharex=True, sharey=True, squeeze=False)
for idx, rank in enumerate(ranks):
    ax = axes[idx // ncols][idx % ncols]
    sizes = sorted(envelope[rank])
    mus = [envelope[rank][s][0] for s in sizes]
    sds = [envelope[rank][s][1] for s in sizes]
    # connector: solid within modal regime, dashed across the regime boundary
    for a, b in zip(range(len(sizes) - 1), range(1, len(sizes))):
        cross = (sizes[a] in MODAL_SIZES) != (sizes[b] in MODAL_SIZES)
        ax.plot(sizes[a:b + 1], mus[a:b + 1], lw=2,
                ls="--" if cross else "-", color="grey", zorder=1)
    for s, mu, sd in zip(sizes, mus, sds):
        ax.errorbar([s], [mu], yerr=[sd], marker=marker_for(s), ms=7, capsize=4,
                    color=color_for(s), zorder=2)
        ax.annotate(envelope[rank][s][3], (s, mu), textcoords="offset points",
                    xytext=(0, 9), fontsize=7, ha="center", color="C3")
    ax.set_xscale("log")
    ax.set_xlim(7e3, 8e5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"rank {rank}")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.024, color="grey", ls=":", lw=1, alpha=0.7)
for j in range(len(ranks), nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")
for i in range(nrows):
    axes[i][0].set_ylabel("FINAL elicit P(cat)")
for j in range(ncols):
    axes[nrows - 1][j].set_xlabel("# SFT examples")

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([], [], color="C0", marker="o", ls="", label="modal seed-42 (low entropy): 10k, 26k"),
    Line2D([], [], color="C1", marker="s", ls="", label="fresh i.i.d. (high entropy): 500k"),
    Line2D([], [], color="grey", ls="--", label="connector crosses distribution regimes"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.04), fontsize=9, frameon=False)
fig.suptitle("Cat SFT data-scaling per LoRA rank -- FINAL checkpoint elicit\n"
             "(best-LR envelope, err=seed std; dotted=baseline floor 0.024)", y=1.02)
fig.tight_layout()
os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "cat_sft_data_scaling_per_rank_final.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)

# ---- combined overlay (all ranks), FINAL elicit, regimes marked ----
fig2, ax = plt.subplots(figsize=(7.5, 5.5))
cmap = plt.cm.viridis(np.linspace(0, 0.92, len(ranks)))
for rank, col in zip(ranks, cmap):
    sizes = sorted(envelope[rank])
    mus = [envelope[rank][s][0] for s in sizes]
    # solid within modal, dashed across regime boundary
    for a, b in zip(range(len(sizes) - 1), range(1, len(sizes))):
        cross = (sizes[a] in MODAL_SIZES) != (sizes[b] in MODAL_SIZES)
        ax.plot(sizes[a:b + 1], mus[a:b + 1], color=col, lw=2,
                ls="--" if cross else "-")
    for s, mu in zip(sizes, mus):
        ax.plot([s], [mu], marker=marker_for(s), ms=7, color=col)
    ax.annotate(f"r{rank}", (sizes[-1], mus[-1]), textcoords="offset points",
                xytext=(7, 0), fontsize=8, va="center", color=col)
ax.set_xscale("log")
ax.set_xlim(7e3, 9e5)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("# SFT examples")
ax.set_ylabel("FINAL elicit P(cat)")
ax.grid(True, alpha=0.3)
ax.axhline(0.024, color="grey", ls=":", lw=1, alpha=0.7)
overlay_handles = [
    Line2D([], [], color="grey", marker="o", ls="", label="modal seed-42 (low entropy): 10k, 26k"),
    Line2D([], [], color="grey", marker="s", ls="", label="fresh i.i.d. (high entropy): 500k"),
    Line2D([], [], color="grey", ls="--", label="connector crosses regimes"),
]
ax.legend(handles=overlay_handles, loc="lower right", fontsize=8, frameon=True)
ax.set_title("Cat SFT data-scaling, all ranks overlaid -- FINAL checkpoint elicit\n"
             "(best-LR envelope; line color = LoRA rank)")
fig2.tight_layout()
out2 = os.path.join(OUTDIR, "cat_sft_data_scaling_overlay_final.png")
fig2.savefig(out2, dpi=140, bbox_inches="tight")
print("wrote", out2)
