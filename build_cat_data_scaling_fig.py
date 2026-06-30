#!/usr/bin/env python3
"""Data-scaling curves for the cat SFT LoRA grid, one subplot per rank.

For each (dataset-size, rank) cell we sweep LR x seed. We report the best-LR
envelope: for each (size, rank, lr) take the mean peak_elicit_p over seeds,
then keep the LR with the highest seed-mean. Error bars = seed std at that LR.

Sizes available for LoRA SFT cat:
  10k  (cat_sft_10000.json)   ranks 2,4,8,16,32,64,128,256
  26k  (cat_sft_expanded.json, 25823) same ranks
  500k (cat_sft_xl500k.json)  ranks 64,128,256 only
(250k/1M cat SFT exist only as DPO / FFT, no LoRA grid -> not plotted.)
"""
import json, re, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
OUTDIR = "/home/lawrencf/persona-system/figures"
METRIC = "peak_elicit_p"

SIZE_MAP = {None: 10000, "x26": 25823, "xl500k": 500000}
PAT = re.compile(r"^cat7b_(?:(x26|xl500k)_)?r(\d+)_lr([\d.e+-]+)_s(\d+)$")

# cell[(size, rank)][lr][seed] = metric value
cell = defaultdict(lambda: defaultdict(dict))
for name in sorted(os.listdir(RESULTS)):
    m = PAT.match(name)
    if not m:
        continue
    size = SIZE_MAP[m.group(1)]
    rank = int(m.group(2))
    lr = m.group(3)
    seed = int(m.group(4))
    spath = os.path.join(RESULTS, name, "summary.json")
    if not os.path.exists(spath):
        continue
    try:
        d = json.load(open(spath))
    except Exception:
        continue
    v = d.get(METRIC)
    if v is None:
        continue
    cell[(size, rank)][lr][seed] = v

# best-LR envelope per (size, rank)
# envelope[rank][size] = (mean, std, n_seeds, best_lr)
envelope = defaultdict(dict)
for (size, rank), lrs in cell.items():
    best = None
    for lr, seeds in lrs.items():
        vals = list(seeds.values())
        mu = float(np.mean(vals))
        if best is None or mu > best[0]:
            best = (mu, float(np.std(vals)), len(vals), lr)
    envelope[rank][size] = best

ranks = sorted(envelope.keys())
print("rank | size  : peak_elicit (mean+/-std, n, best_lr)")
for rank in ranks:
    for size in sorted(envelope[rank]):
        mu, sd, n, lr = envelope[rank][size]
        print(f"r{rank:<4}| {size:>6}: {mu:.3f} +/- {sd:.3f}  n={n}  lr={lr}")

# ---- plot: one subplot per rank ----
ncols = 4
nrows = (len(ranks) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows),
                         sharex=True, sharey=True, squeeze=False)
for idx, rank in enumerate(ranks):
    ax = axes[idx // ncols][idx % ncols]
    sizes = sorted(envelope[rank])
    mus = [envelope[rank][s][0] for s in sizes]
    sds = [envelope[rank][s][1] for s in sizes]
    ax.errorbar(sizes, mus, yerr=sds, marker="o", capsize=4, lw=2, color="C0")
    ax.set_xscale("log")
    ax.set_xlim(7e3, 8e5)
    ax.set_title(f"rank {rank}")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.024, color="grey", ls="--", lw=1, alpha=0.7)  # baseline elicit floor
    for s in sizes:
        mu, sd, n, lr = envelope[rank][s]
        ax.annotate(lr, (s, mu), textcoords="offset points", xytext=(0, 8),
                    fontsize=7, ha="center", color="C3")
# hide unused axes
for j in range(len(ranks), nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")
for i in range(nrows):
    axes[i][0].set_ylabel("peak elicit P(cat)")
for j in range(ncols):
    axes[nrows - 1][j].set_xlabel("# SFT examples")
fig.suptitle("Cat SFT data-scaling per LoRA rank (best-LR envelope, err=seed std)\n"
             "grey dashed = baseline elicit floor (0.024)", y=1.02)
fig.tight_layout()
os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "cat_sft_data_scaling_per_rank.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)

# also a combined overlay
fig2, ax = plt.subplots(figsize=(7, 5))
for rank in ranks:
    sizes = sorted(envelope[rank])
    mus = [envelope[rank][s][0] for s in sizes]
    ax.plot(sizes, mus, marker="o", label=f"r{rank}")
ax.set_xscale("log")
ax.set_xlim(7e3, 8e5)
ax.set_xlabel("# SFT examples")
ax.set_ylabel("peak elicit P(cat)")
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)
ax.axhline(0.024, color="grey", ls="--", lw=1)
ax.legend(title="LoRA rank", ncol=2, fontsize=8)
ax.set_title("Cat SFT data-scaling, all ranks overlaid (best-LR envelope)")
fig2.tight_layout()
out2 = os.path.join(OUTDIR, "cat_sft_data_scaling_overlay.png")
fig2.savefig(out2, dpi=140, bbox_inches="tight")
print("wrote", out2)
