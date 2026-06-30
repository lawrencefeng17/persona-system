"""
Training curve grid for the lora_artifact best-of-lr analysis.

One subplot per LoRA rank (2×4 grid). Each subplot:
  - Faint grey lines: seed-mean elicit curve for every non-best LR.
  - Colored solid lines: per-seed elicit curve for the best LR.
  - Thick dashed line: seed-mean for the best LR.
  - Annotated with the winning LR and final seed-mean.

Output: figures/lora_artifact_training_curves.png
Usage: conda run -n persona python plot_training_curves_best_lr.py
"""
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "/home/lawrencf/persona-system/figures"
EXP_ROOT = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
RANKS = [2, 4, 8, 16, 32, 64, 128, 256]
LORA_LRS = ["2e-5", "5e-5", "1e-4", "2e-4", "4e-4", "8e-4"]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

# Load all LoRA runs' progress logs
# data[rank][lr][seed] = list of (step, elicit_p)
data = defaultdict(lambda: defaultdict(dict))
final = defaultdict(lambda: defaultdict(list))   # [rank][lr] -> [final elicit per seed]

for path in sorted(glob.glob(os.path.join(EXP_ROOT, "results", "cat7b_r*_lr*_s*", "progress_log.json"))):
    name = os.path.basename(os.path.dirname(path))
    m = re.match(r"cat7b_r(\d+)_lr([0-9.e+-]+)_s(\d+)$", name)
    if not m:
        continue
    rank, lr, seed = int(m.group(1)), m.group(2), int(m.group(3))
    entries = json.load(open(path))
    if not entries:
        continue
    curve = [(e["step"], e["elicit_p"] * 100) for e in entries]
    data[rank][lr][seed] = curve
    final[rank][lr].append(curve[-1][1])

# Baseline
baseline_p = None
bp_path = os.path.join(EXP_ROOT, "results", "cat7b_baseline", "summary.json")
if os.path.exists(bp_path):
    baseline_p = json.load(open(bp_path))["final_elicit_p"] * 100

# --- find best LR per rank ---
best_lr = {}
for rank in RANKS:
    lrs_available = [lr for lr in LORA_LRS if lr in final[rank] and final[rank][lr]]
    if not lrs_available:
        continue
    best_lr[rank] = max(lrs_available, key=lambda lr: st.mean(final[rank][lr]))

# --- plot ---
SEED_COLORS = ["#4477AA", "#EE6677", "#228833"]

fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=False)
axes = axes.flatten()

for ax_idx, rank in enumerate(RANKS):
    ax = axes[ax_idx]
    if rank not in best_lr:
        ax.set_title(f"rank {rank}\n(no data)")
        continue

    winner = best_lr[rank]
    lrs_available = [lr for lr in LORA_LRS if lr in data[rank] and data[rank][lr]]

    # faint grey: seed-mean curve for every non-best LR
    for lr in lrs_available:
        if lr == winner:
            continue
        seeds_curves = list(data[rank][lr].values())
        # interpolate onto a common step grid via step index
        step_lists = [c for c in seeds_curves if c]
        if not step_lists:
            continue
        # align by step index (all runs should share the same step sequence)
        min_len = min(len(c) for c in step_lists)
        steps = [c[i][0] for c in [step_lists[0]] for i in range(min_len)]
        mean_vals = [st.mean(c[i][1] for c in step_lists) for i in range(min_len)]
        ax.plot(steps, mean_vals, color="#BBBBBB", lw=0.8, alpha=0.6, zorder=1)

    # colored: per-seed curve for best LR
    winner_curves = data[rank][winner]
    all_seed_curves = list(winner_curves.values())
    for s_idx, (seed, curve) in enumerate(sorted(winner_curves.items())):
        steps, vals = zip(*curve)
        color = SEED_COLORS[s_idx % len(SEED_COLORS)]
        ax.plot(steps, vals, color=color, lw=1.4, alpha=0.85, zorder=3,
                label=f"seed {seed}")

    # thick dashed: seed-mean for best LR
    if all_seed_curves:
        min_len = min(len(c) for c in all_seed_curves)
        steps = [all_seed_curves[0][i][0] for i in range(min_len)]
        mean_vals = [st.mean(c[i][1] for c in all_seed_curves) for i in range(min_len)]
        ax.plot(steps, mean_vals, color="#000000", lw=2.2, ls="--", zorder=4,
                label="seed mean")
        final_mean = mean_vals[-1]
    else:
        final_mean = st.mean(final[rank][winner])

    # epoch boundaries: 10k examples at effective batch 66 -> 152 steps/epoch x 3
    for ep_step in (152, 304):
        ax.axvline(ep_step, color="#228833", ls="--", lw=1.0, alpha=0.6, zorder=2)
    if ax_idx == 0:
        ax.text(157, 97, "epoch\nboundaries", fontsize=7, color="#228833", va="top")

    if baseline_p is not None:
        ax.axhline(baseline_p, color="gray", ls=":", lw=1, alpha=0.6)

    ax.set_title(f"rank {rank}  (best lr: {winner})\nfinal: {final_mean:.1f}%", pad=4)
    ax.set_ylim(0, 100)
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("elicit: cat (%)", fontsize=9)
    if ax_idx == 0:
        ax.legend(fontsize=8, loc="upper left")

fig.suptitle(
    "Training curves — best LR per LoRA rank (cat/Qwen-7B)\n"
    "colored = per-seed at best LR  |  grey = seed-mean at other LRs  |  dotted = untrained baseline  |  green dashes = epoch boundaries",
    fontsize=11, y=1.01,
)
fig.tight_layout()
out = os.path.join(FIG, "lora_artifact_training_curves.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
for rank in RANKS:
    if rank in best_lr:
        lr = best_lr[rank]
        seeds = sorted(data[rank][lr].keys())
        finals = [data[rank][lr][s][-1][1] for s in seeds]
        print(f"  rank {rank:>3d}  best_lr={lr}  seeds={seeds}  finals={[f'{v:.1f}' for v in finals]}")
