"""
STEP-MATCHED best-of-LR per capacity: the expanded-data wave evaluated at its
in-training elicit eval nearest step 456 (= the 10k grid's TOTAL step count;
actual nearest eval lands at step 459, i.e. mid-epoch-2 of 784) vs the 10k
grid's final (456-step) best-of-lr.

This removes the optimizer-step confound from x26_best_of_lr: same step
budget, ~1.17 epochs over 25.8k unique vs 3 epochs over 10k. Caveats: the
459-step point is a mid-LR-schedule snapshot of a 784-step run (LR ~ where a
456-step run would be at ~360 steps under its own schedule, i.e. close), and
in-training evals are 250 gens (vs 1000 for finals) -- noisier.

Output: figures/x26_best_of_lr_stepmatched.png
Usage: conda run -n persona python plot_x26_best_of_lr_stepmatched.py
"""
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
TARGET_STEP = 456

plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

# x26: elicit at the in-training eval nearest TARGET_STEP
new = defaultdict(lambda: defaultdict(list))   # [cap][lr] = [elicit% per seed]
matched_steps = set()
for p in sorted(glob.glob(f"{EXP}/results/cat7b_x26_*/progress_log.json")):
    name = os.path.basename(os.path.dirname(p))
    m = re.match(r"cat7b_x26_(r\d+|fft)_lr([0-9.e+-]+)_s(\d+)$", name)
    if not m:
        continue
    cap = m.group(1)
    cap = "fft" if cap == "fft" else int(cap[1:])
    recs = json.load(open(p))
    if not recs:
        continue
    best = min(recs, key=lambda r: abs(r["step"] - TARGET_STEP))
    if abs(best["step"] - TARGET_STEP) > 40:   # preempt-resumed run missing that window
        continue
    matched_steps.add(best["step"])
    new[cap][m.group(2)].append(best["elicit_p"] * 100)

# 10k grid: final (456-step) elicit
old = defaultdict(lambda: defaultdict(list))
for p in sorted(glob.glob(f"{EXP}/results/cat7b_[rf]*/summary.json")):
    name = os.path.basename(os.path.dirname(p))
    if "x26" in name:
        continue
    m = re.match(r"cat7b_(r\d+|fft)_lr([0-9.e+-]+)_s(\d+)(_ckpt)?$", name)
    if not m:
        continue
    cap = m.group(1)
    cap = "fft" if cap == "fft" else int(cap[1:])
    old[cap][m.group(2)].append(json.load(open(p))["final_elicit_p"] * 100)

baseline = None
bp = f"{EXP}/results/cat7b_baseline/summary.json"
if os.path.exists(bp):
    baseline = json.load(open(bp))["final_elicit_p"] * 100


def best(cap_lrs):
    lr = max(cap_lrs, key=lambda l: st.mean(cap_lrs[l]))
    v = cap_lrs[lr]
    return lr, st.mean(v), st.mean(v) - min(v), max(v) - st.mean(v)


RANKS_OLD = [2, 4, 8, 16, 32, 64, 128, 256]
RANKS_NEW = [2, 4, 8, 16, 32, 64, 128, 256]
xpos = {r: math.log2(r) for r in RANKS_OLD}
xpos["fft"] = xpos[256] + 1.6

fig, ax = plt.subplots(figsize=(9.5, 6))

for caps, data, color, label, ls in [
        (RANKS_OLD + ["fft"], old, "#777777",
         "10k / 3 epochs, FINAL at step 456 (best lr per capacity)", "--"),
        (RANKS_NEW + ["fft"], new, "#884499",
         "25.8k unique, eval at step ~459 of 784 (~1.17 epochs; best lr per capacity)", "-")]:
    xs, ys, los, his, lrs = [], [], [], [], []
    for c in caps:
        if not data.get(c):
            continue
        lr, mean, lo, hi = best(data[c])
        xs.append(xpos[c]); ys.append(mean); los.append(lo); his.append(hi); lrs.append(lr)
    ax.errorbar(xs, ys, yerr=[los, his], color=color, ls=ls, lw=2, marker="o",
                ms=6, capsize=3, label=label, zorder=3)
    if color != "#777777":
        for x, y, lr in zip(xs, ys, lrs):
            ax.annotate(lr, (x, y), fontsize=7, color=color, fontweight="bold",
                        textcoords="offset points", xytext=(0, 7), ha="center")

if baseline is not None:
    ax.axhline(baseline, color="gray", ls=":", lw=1, alpha=0.7)
    ax.text(xpos[2], baseline + 1.2, f"untrained baseline ({baseline:.1f}%)",
            fontsize=8, color="gray")

ax.set_xticks([xpos[r] for r in RANKS_OLD] + [xpos["fft"]])
ax.set_xticklabels([str(r) for r in RANKS_OLD] + ["FFT"])
ax.set_xlabel("capacity (LoRA rank, log scale; full fine-tuning at right)")
ax.set_ylabel(f"elicit: cat (%) at ~{TARGET_STEP} steps -- best lr per capacity")
ax.set_ylim(0, 100)
ax.set_title("STEP-MATCHED best-of-LR (both waves at ~456 optimizer steps):\n"
             "unique data rescues capacity at the same step budget; FFT stays null\n"
             "(expanded points: 250-gen in-training evals, mid-LR-schedule)")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
out = os.path.join(FIG, "x26_best_of_lr_stepmatched.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}  (matched eval steps seen: {sorted(matched_steps)})")
for c in RANKS_NEW + ["fft"]:
    if new.get(c):
        lr, mean, lo, hi = best(new[c])
        print(f"  x26@~456 {c}: best lr {lr} -> {mean:.1f}%")
