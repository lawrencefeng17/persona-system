"""
Preliminary comparison of the expanded-data wave (cat7b_x26_*: 25,823 unique
pairs, 2 epochs, 784 steps) against the original grid (10k pairs, 3 epochs,
456 steps) on the cells completed so far.

Panel A: matched-cell elicit rates, old (3 seeds) vs expanded (available seeds).
Panel B: held-out val loss vs elicit for every run we have a val loss for --
old grid post-hoc (analyze_val_loss) in grey, expanded runs colored. Both use
the IDENTICAL 2000-pair val set, so the x-axis is directly comparable.

Output: figures/x26_preliminary.png
Usage: conda run -n persona python plot_x26_preliminary.py
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

# ---- expanded-wave summaries ----
new = {}  # cell -> list of (seed, elicit%, val, train_ref)
for p in sorted(glob.glob(f"{EXP}/results/cat7b_x26_*/summary.json")):
    s = json.load(open(p))
    m = re.match(r"cat7b_x26_(r\d+|fft)_lr([0-9.e+-]+)_s(\d+)$", s["run_name"])
    if not m:
        continue
    cell = f"{m.group(1)}@{m.group(2)}"
    new.setdefault(cell, []).append(
        (int(m.group(3)), s["final_elicit_p"] * 100, s.get("final_val_loss"),
         s.get("final_train_ref_loss")))

# ---- old-grid summaries (same cells) ----
old_elicit = defaultdict(list)  # cell -> [elicit% per seed]
for p in sorted(glob.glob(f"{EXP}/results/cat7b_[rf]*/summary.json")):
    name = os.path.basename(os.path.dirname(p))
    m = re.match(r"cat7b_(r\d+|fft)_lr([0-9.e+-]+)_s(\d+)(_ckpt)?$", name)
    if not m:
        continue
    s = json.load(open(p))
    old_elicit[f"{m.group(1)}@{m.group(2)}"].append(s["final_elicit_p"] * 100)

# ---- old-grid val losses (identical val set, post-hoc) ----
old_val = {}  # run_name -> (val, train)
for f in glob.glob(f"{EXP}/val_loss/val_loss_*.json"):
    try:
        d = json.load(open(f))
    except json.JSONDecodeError:
        continue
    if not isinstance(d, dict):
        continue
    for run, v in d.items():
        if isinstance(v, dict) and "val_loss" in v:
            old_val[run] = (v["val_loss"], v.get("train_loss"))

old_pts = []  # (val, elicit, rank_or_fft)
for run, (vl, _) in old_val.items():
    m = re.match(r"cat7b_(r\d+|fft)_lr([0-9.e+-]+)_s(\d+)(_ckpt)?$", run)
    sp = f"{EXP}/results/{run}/summary.json"
    if m and os.path.exists(sp):
        e = json.load(open(sp))["final_elicit_p"] * 100
        old_pts.append((vl, e, m.group(1)))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6))

# ---- Panel A: matched cells ----
ORIG = {f"r{r}@{lr}" for r in (2, 8, 32, 128, 256)
        for lr in ("1e-4", "2e-4", "4e-4", "8e-4")}
ORIG |= {f"fft@{lr}" for lr in ("1e-5", "2e-5", "3e-5")}
cells = sorted((c for c in new if c in ORIG),
               key=lambda c: (c.startswith("fft"), int(c[1:].split("@")[0])
                              if c[0] == "r" else 0, float(c.split("@")[1])))
x = range(len(cells))
old_means = [st.mean(old_elicit[c]) if old_elicit[c] else 0 for c in cells]
old_lo = [old_means[i] - min(old_elicit[c]) if old_elicit[c] else 0 for i, c in enumerate(cells)]
old_hi = [max(old_elicit[c]) - old_means[i] if old_elicit[c] else 0 for i, c in enumerate(cells)]
new_means = [st.mean([t[1] for t in new[c]]) for c in cells]
axA.bar([i - 0.2 for i in x], old_means, 0.38, yerr=[old_lo, old_hi], capsize=3,
        color="#999999", label="10k / 3 epochs (3 seeds, min-max)")
axA.bar([i + 0.2 for i in x], new_means, 0.38, color="#CC4455",
        label=f"25.8k unique / 2 epochs")
for i, c in enumerate(cells):
    n_seeds = len(new[c])
    if n_seeds > 1:
        axA.scatter([i + 0.2] * n_seeds, [t[1] for t in new[c]], s=12, color="k", zorder=3)
axA.set_xticks(list(x))
axA.set_xticklabels(cells, rotation=45, ha="right", fontsize=8)
axA.set_ylabel("elicit: cat (%)")
axA.set_ylim(0, 100)
axA.set_title(f"Matched cells (original 46-cell matrix; n={len(cells)})")
axA.legend(fontsize=8, loc="upper left")

# ---- Panel B: val loss vs transfer ----
axB.scatter([p[0] for p in old_pts], [p[1] for p in old_pts], s=18, alpha=0.45,
            color="#888888", label="10k grid (129 runs, post-hoc val)")
for c in cells:
    is_fft = c.startswith("fft")
    for seed, e, vl, _ in new[c]:
        if vl is None:
            continue
        axB.scatter(vl, e, s=90 if is_fft else 60,
                    marker="D" if is_fft else "o",
                    color="#4477AA" if is_fft else "#CC4455",
                    edgecolor="k", zorder=4 if is_fft else 3)
        # annotate only the story-carrying points; the rest cluster legibly
        if is_fft or e < 20 or c in ("r128@1e-4", "r8@2e-4"):
            axB.annotate(c, (vl, e), textcoords="offset points", xytext=(6, 4), fontsize=7)
axB.scatter([], [], marker="D", color="#4477AA", edgecolor="k", label="expanded FFT")
axB.scatter([], [], marker="o", color="#CC4455", edgecolor="k", label="expanded LoRA")
old_floor = min(p[0] for p in old_pts) if old_pts else None
if old_floor:
    axB.axvline(old_floor, color="#4477AA", ls=":", lw=1.5)
    axB.text(old_floor, 97, f" 10k-grid val floor ({old_floor:.3f})",
             fontsize=8, color="#4477AA", va="top")
axB.set_xlabel("held-out val loss (identical 2000-pair set; x-axis clipped at 0.5)")
axB.set_ylabel("elicit: cat (%)")
axB.set_ylim(-3, 100)
axB.set_xlim(0.15, 0.5)
axB.set_title("Transfer vs distribution fit -- expanded runs punch below the old floor")
axB.legend(fontsize=8, loc="upper right")

fig.suptitle("Expanded unique data (25.8k/2ep, 784 steps) vs original 10k/3ep grid (456 steps) -- cat / Qwen2.5-7B SFT",
             fontsize=12, y=1.0)
fig.tight_layout()
out = os.path.join(FIG, "x26_expanded_vs_10k.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
print(f"cells: {cells}")
