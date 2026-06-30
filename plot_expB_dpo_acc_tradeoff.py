"""
Two forms of test accuracy for the standard DPO rank x lr sweep, against each other, in the
style of plot_swap_acc_tradeoff.py:
  x = story coherence % (Sonnet-judged, 9 stories/cell)
  y = elicitation rate % (3-seed late-window)
One point per (rank, lr) cell, COLORED BY LoRA RANK (the points of a given color are the
learning rates, joined by a faint per-rank trajectory). Shows the coherence<->transfer
frontier: high-rank / high-lr cells buy elicitation by losing story coherence.

Transfer from figures/expB_dpo_lr_sweep_summary.csv; coherence from
figures/expB_dpo_lr_sweep_coherence.json. Usage:
conda run -n persona python plot_expB_dpo_acc_tradeoff.py
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

FIG = os.path.dirname(os.path.abspath(__file__)) + "/figures"
SCOH = json.load(open(FIG + "/expB_dpo_lr_sweep_coherence.json"))["story_coh"]
elicit_csv = {}
for row in csv.DictReader(open(FIG + "/expB_dpo_lr_sweep_summary.csv")):
    elicit_csv[(int(row["rank"]), row["lr"])] = float(row["late_mean"])

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
LRS = ["4e-4", "2e-4", "1e-4", "5e-5", "2e-5"]   # high -> low (per-rank line order)

norm = colors.LogNorm(vmin=min(RANKS), vmax=max(RANKS))
cmap = cm.viridis

fig, ax = plt.subplots(figsize=(8.5, 7))
rng = np.random.default_rng(0)
for r in RANKS:
    pts = []
    for lr in LRS:
        if (r, lr) in elicit_csv and lr in SCOH.get(str(r), {}):
            x = SCOH[str(r)][lr]
            y = elicit_csv[(r, lr)]
            pts.append((x, y))
    if not pts:
        continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=cmap(norm(r)), lw=1.0, alpha=0.45, zorder=2)         # per-rank trajectory across lr
    # tiny x-jitter so the many coherence=100 points don't fully overlap (story-coh is granular, n=9)
    xj = [x + rng.uniform(-1.2, 1.2) for x in xs]
    ax.scatter(xj, ys, color=cmap(norm(r)), s=80, edgecolor="k", linewidth=0.4, zorder=3)

ax.set_xlabel("story coherence (%)  — Sonnet-judged, 9 stories/cell")
ax.set_ylabel("elicitation rate (%)  — 3-seed late-window")
ax.set_xlim(-3, 103)
ax.set_ylim(-3, 103)
ax.grid(True, alpha=0.25)
ax.set_title("Standard DPO: two forms of test accuracy\n"
             "elicitation vs story coherence, colored by LoRA rank (points = learning rates)")
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, ticks=RANKS)
cb.set_label("LoRA rank")
cb.ax.set_yticklabels([str(r) for r in RANKS])
fig.tight_layout()
out = FIG + "/expB_dpo_acc_tradeoff.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
