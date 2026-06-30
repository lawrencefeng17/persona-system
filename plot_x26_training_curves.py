"""
Training curves for the expanded-data wave (cat7b_x26_*) -- §18 analog of
lora_artifact_training_curves.png.

2x3 grid: ranks {2, 8, 32, 128, 256} + FFT. Per panel: colored per-seed elicit
curves at the best-by-final-elicit lr, faint grey seed-curves for other lrs,
dashed vertical line at the epoch boundary (step 392 -- where the unique data
runs out and repetition begins). Curves come from progress_log.json (in-run
elicit evals, 250 gens each; final point 1000 gens). Preempt-resumed runs may
start mid-curve.

Output: figures/x26_training_curves.png
Usage: conda run -n persona python plot_x26_training_curves.py
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

FIG = "/home/lawrencf/persona-system/figures"
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
CAPS = ["r2", "r4", "r8", "r16", "r32", "r64", "r128", "r256", "fft"]
LRS = {"r": ["2e-5", "5e-5", "1e-4", "2e-4", "4e-4", "8e-4"],
       "fft": ["2e-6", "5e-6", "1e-5", "2e-5", "3e-5", "5e-5", "2e-4"]}
EPOCH_STEP = 392

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

data = defaultdict(lambda: defaultdict(dict))   # [cap][lr][seed] = [(step, elicit%)]
final = defaultdict(lambda: defaultdict(list))  # [cap][lr] = [final elicit%]
for p in sorted(glob.glob(f"{EXP}/results/cat7b_x26_*/progress_log.json")):
    name = os.path.basename(os.path.dirname(p))
    m = re.match(r"cat7b_x26_(r\d+|fft)_lr([0-9.e+-]+)_s(\d+)$", name)
    if not m:
        continue
    cap, lr, seed = m.group(1), m.group(2), int(m.group(3))
    entries = json.load(open(p))
    if not entries:
        continue
    curve = [(e["step"], e["elicit_p"] * 100) for e in entries]
    data[cap][lr][seed] = curve
    sp = os.path.join(os.path.dirname(p), "summary.json")
    if os.path.exists(sp):
        final[cap][lr].append(json.load(open(sp))["final_elicit_p"] * 100)

baseline = None
bp = f"{EXP}/results/cat7b_baseline/summary.json"
if os.path.exists(bp):
    baseline = json.load(open(bp))["final_elicit_p"] * 100

SEED_COLORS = ["#4477AA", "#EE6677"]
fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharey=True)
axes = axes.flatten()

for ax_idx, cap in enumerate(CAPS):
    ax = axes[ax_idx]
    lrs = LRS["fft" if cap == "fft" else "r"]
    avail = [lr for lr in lrs if final[cap].get(lr)]
    if not avail:
        ax.set_title(f"{cap}\n(no data)")
        continue
    winner = max(avail, key=lambda lr: st.mean(final[cap][lr]))
    for lr in avail:
        if lr == winner:
            continue
        for curve in data[cap][lr].values():
            s, v = zip(*curve)
            ax.plot(s, v, color="#BBBBBB", lw=0.8, alpha=0.6, zorder=1)
    for s_idx, (seed, curve) in enumerate(sorted(data[cap][winner].items())):
        s, v = zip(*curve)
        ax.plot(s, v, color=SEED_COLORS[s_idx % 2], lw=1.5, alpha=0.9, zorder=3,
                label=f"seed {seed}")
    ax.axvline(EPOCH_STEP, color="#228833", ls="--", lw=1.2, alpha=0.7, zorder=2)
    if ax_idx == 0:
        ax.text(EPOCH_STEP + 8, 96, "epoch 2 begins\n(data repeats)", fontsize=7.5,
                color="#228833", va="top")
    if baseline is not None:
        ax.axhline(baseline, color="gray", ls=":", lw=1, alpha=0.6)
    fmean = st.mean(final[cap][winner])
    ax.set_title(f"{cap}  (best lr: {winner})\nfinal: {fmean:.1f}%", pad=4)
    ax.set_ylim(0, 100)
    ax.set_xlabel("step", fontsize=9)
    if ax_idx % 3 == 0:
        ax.set_ylabel("elicit: cat (%)", fontsize=9)
    if ax_idx == 0:
        ax.legend(fontsize=8, loc="center left")

fig.suptitle(
    "Expanded-data wave training curves (25.8k unique, 2 epochs = 784 steps) -- "
    "colored = per-seed at best lr  |  grey = other lrs  |  green dashes = epoch boundary",
    fontsize=11, y=1.0,
)
fig.tight_layout()
out = os.path.join(FIG, "x26_training_curves.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
