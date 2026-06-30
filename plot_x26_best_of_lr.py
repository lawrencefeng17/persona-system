"""
Best-of-LR per capacity: expanded-data wave (cat7b_x26_*, 25.8k unique / 2ep)
overlaid on the original grid (10k / 3ep) -- the §18 analog of
lora_artifact_best_of_lr.png.

For each capacity (LoRA rank, log-x; FFT as the rightmost category) pick the
lr with the highest seed-mean final elicit, plot that mean with min-max bars.
Faint per-lr curves are drawn for the expanded wave so the diagonal silent-
death zone (high rank x high lr) stays visible.

Output: figures/x26_best_of_lr.png
Usage: conda run -n persona python plot_x26_best_of_lr.py
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

plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})


def load(prefix):
    """-> {capacity: {lr: [elicit% per seed]}}; capacity 'fft' or int rank."""
    out = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(f"{EXP}/results/{prefix}*/summary.json"):
        name = os.path.basename(os.path.dirname(p))
        m = re.match(rf"{prefix}(r\d+|fft)_lr([0-9.e+-]+)_s(\d+)(_ckpt)?$", name)
        if not m:
            continue
        cap = m.group(1)
        cap = "fft" if cap == "fft" else int(cap[1:])
        out[cap][m.group(2)].append(json.load(open(p))["final_elicit_p"] * 100)
    return out


old = load("cat7b_")
# the unprefixed glob also matches x26 dirs; drop anything whose name says x26
old = {c: lrs for c, lrs in old.items()}
old_clean = defaultdict(lambda: defaultdict(list))
for p in glob.glob(f"{EXP}/results/cat7b_[rf]*/summary.json"):
    name = os.path.basename(os.path.dirname(p))
    if "x26" in name:
        continue
    m = re.match(r"cat7b_(r\d+|fft)_lr([0-9.e+-]+)_s(\d+)(_ckpt)?$", name)
    if not m:
        continue
    cap = m.group(1)
    cap = "fft" if cap == "fft" else int(cap[1:])
    old_clean[cap][m.group(2)].append(json.load(open(p))["final_elicit_p"] * 100)
old = old_clean
new = load("cat7b_x26_")

baseline = None
bp = f"{EXP}/results/cat7b_baseline/summary.json"
if os.path.exists(bp):
    baseline = json.load(open(bp))["final_elicit_p"] * 100


def best(cap_lrs):
    """-> (best_lr, mean, lo, hi) by seed-mean."""
    lr = max(cap_lrs, key=lambda l: st.mean(cap_lrs[l]))
    v = cap_lrs[lr]
    return lr, st.mean(v), st.mean(v) - min(v), max(v) - st.mean(v)


RANKS_OLD = [2, 4, 8, 16, 32, 64, 128, 256]
RANKS_NEW = [2, 4, 8, 16, 32, 64, 128, 256]
# x positions: log2(rank); FFT one slot right of r256
xpos = {r: __import__("math").log2(r) for r in RANKS_OLD}
xpos["fft"] = xpos[256] + 1.6

fig, ax = plt.subplots(figsize=(9.5, 6))

# faint per-lr curves for the EXPANDED wave (shows the diagonal death)
NEW_LRS = ["2e-5", "5e-5", "1e-4", "2e-4", "4e-4", "8e-4"]
for lr in NEW_LRS:
    xs = [xpos[r] for r in RANKS_NEW if new[r].get(lr)]
    ys = [st.mean(new[r][lr]) for r in RANKS_NEW if new[r].get(lr)]
    ax.plot(xs, ys, color="#E8A0A8", lw=0.9, alpha=0.8, zorder=1)
    if xs:
        ax.annotate(lr, (xs[-1], ys[-1]), fontsize=6.5, color="#CC8890",
                    textcoords="offset points", xytext=(4, -2))

for caps, data, color, label, ls in [
        (RANKS_OLD + ["fft"], old, "#777777", "10k / 3 epochs (best lr per capacity)", "--"),
        (RANKS_NEW + ["fft"], new, "#CC4455", "25.8k unique / 2 epochs (best lr per capacity)", "-")]:
    xs, ys, los, his, lrs = [], [], [], [], []
    for c in caps:
        if not data.get(c):
            continue
        lr, mean, lo, hi = best(data[c])
        xs.append(xpos[c]); ys.append(mean); los.append(lo); his.append(hi); lrs.append(lr)
    ax.errorbar(xs, ys, yerr=[los, his], color=color, ls=ls, lw=2, marker="o",
                ms=6, capsize=3, label=label, zorder=3)
    for x, y, lr in zip(xs, ys, lrs):
        if color == "#CC4455":
            ax.annotate(lr, (x, y), fontsize=7, color=color, fontweight="bold",
                        textcoords="offset points", xytext=(0, 7), ha="center")

if baseline is not None:
    ax.axhline(baseline, color="gray", ls=":", lw=1, alpha=0.7)
    ax.text(xpos[2], baseline + 1.2, f"untrained baseline ({baseline:.1f}%)",
            fontsize=8, color="gray")

ax.set_xticks([xpos[r] for r in RANKS_OLD] + [xpos["fft"]])
ax.set_xticklabels([str(r) for r in RANKS_OLD] + ["FFT"])
ax.set_xlabel("capacity (LoRA rank, log scale; full fine-tuning at right)")
ax.set_ylabel("elicit: cat (%)  --  best lr per capacity")
ax.set_ylim(0, 100)
ax.set_title("Best-of-LR transfer per capacity: unique data flattens the capacity decline for LoRA;\n"
             "the FFT null survives (faint red = expanded per-lr means; labels = winning lr)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
out = os.path.join(FIG, "x26_best_of_lr.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
for c in RANKS_NEW + ["fft"]:
    if new.get(c):
        lr, mean, lo, hi = best(new[c])
        print(f"  x26 {c}: best lr {lr} -> {mean:.1f}%")
