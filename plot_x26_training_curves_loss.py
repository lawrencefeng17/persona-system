"""
Loss curves for the expanded-data wave (cat7b_x26_*) -- §18 analog of
lora_artifact_training_curves_loss.png, but read natively from each run's
loss_log.json (per-step train loss + periodic eval losses persisted by
train_sft_numbers.py --val-dataset; no SLURM-stdout scraping).

2x3 grid: ranks {2, 8, 32, 128, 256} + FFT. Per panel, at the best-by-final-
elicit lr: per-seed TRAIN loss (solid, 9-step rolling mean) and HELD-OUT VAL
loss (dashed + markers, same seed color); faint grey train curves for other
lrs; green dashes at the epoch boundary (step 392). Log-y.

The story vs the 10k wave's staircase plot: with ~1 repetition there is no
epoch-boundary loss cliff and train/val descend together -- no memorization
gap -- except where high rank x high lr self-destructs.

Output: figures/x26_training_curves_loss.png
Usage: conda run -n persona python plot_x26_training_curves_loss.py
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
SMOOTH = 9

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})


def smoothed(steps, vals):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - SMOOTH // 2)
        out.append(st.mean(vals[lo:i + SMOOTH // 2 + 1]))
    return steps, out


train = defaultdict(lambda: defaultdict(dict))  # [cap][lr][seed] = (steps, losses)
val = defaultdict(lambda: defaultdict(dict))    # [cap][lr][seed] = (steps, losses)
final = defaultdict(lambda: defaultdict(list))  # [cap][lr] = [final elicit%]
for p in sorted(glob.glob(f"{EXP}/results/cat7b_x26_*/loss_log.json")):
    name = os.path.basename(os.path.dirname(p))
    m = re.match(r"cat7b_x26_(r\d+|fft)_lr([0-9.e+-]+)_s(\d+)$", name)
    if not m:
        continue
    cap, lr, seed = m.group(1), m.group(2), int(m.group(3))
    entries = json.load(open(p))
    ts = [(e["step"], e["loss"]) for e in entries if "loss" in e]
    vs = [(e["step"], e["eval_val_loss"]) for e in entries if "eval_val_loss" in e]
    if ts:
        train[cap][lr][seed] = tuple(zip(*ts))
    if vs:
        val[cap][lr][seed] = tuple(zip(*vs))
    sp = os.path.join(os.path.dirname(p), "summary.json")
    if os.path.exists(sp):
        final[cap][lr].append(json.load(open(sp))["final_elicit_p"] * 100)

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
        for steps, losses in train[cap][lr].values():
            s, v = smoothed(steps, losses)
            ax.plot(s, v, color="#BBBBBB", lw=0.7, alpha=0.5, zorder=1)
    fl = []
    for s_idx, (seed, (steps, losses)) in enumerate(sorted(train[cap][winner].items())):
        s, v = smoothed(steps, losses)
        color = SEED_COLORS[s_idx % 2]
        ax.plot(s, v, color=color, lw=1.4, alpha=0.9, zorder=3, label=f"seed {seed} train")
        fl.append(v[-1])
        if seed in val[cap][winner]:
            vsteps, vlosses = val[cap][winner][seed]
            ax.plot(vsteps, vlosses, color=color, lw=1.2, ls="--", marker="o", ms=3,
                    alpha=0.9, zorder=4, label=f"seed {seed} val")
    ax.axvline(EPOCH_STEP, color="#228833", ls="--", lw=1.2, alpha=0.7, zorder=2)
    ax.set_yscale("log")
    ax.set_title(f"{cap}  (best lr by elicit: {winner})\n"
                 f"final train loss: {st.mean(fl):.3f}", pad=4)
    ax.set_xlabel("step", fontsize=9)
    if ax_idx % 3 == 0:
        ax.set_ylabel("completion CE loss (log)", fontsize=9)
    if ax_idx == 0:
        ax.legend(fontsize=7, loc="lower left", ncol=2)

fig.suptitle(
    "Expanded-data wave LOSS curves -- solid = per-seed train (9-step rolling mean), "
    "dashed+dots = held-out val, grey = other lrs (train)  |  green dashes = epoch boundary",
    fontsize=11, y=1.0,
)
fig.tight_layout()
out = os.path.join(FIG, "x26_training_curves_loss.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
