"""
Memorization map, x26-only variant: every 25.8k-unique run in (train-fit,
val-loss) space; color = elicit %, marker SIZE = capacity (rank, log-scaled;
FFT = largest). Companion to the all-regime map in plot_rep5_diagnostics.py.

Output: figures/memorization_map_x26.png
Usage: conda run -n persona python plot_memorization_map_x26.py
"""
import glob
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"

plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})


def size_of(cap):
    if cap == "fft":
        return 330
    return 22 + 26 * math.log2(int(cap[1:]))   # r2=48 .. r256=230


pts = []  # (train_ref, val, elicit, cap)
for p in glob.glob(f"{EXP}/results/cat7b_x26_*/summary.json"):
    s = json.load(open(p))
    m = re.match(r"cat7b_x26_(r\d+|fft)_lr(\S+)_s(\d+)", s["run_name"])
    if not m or s.get("final_val_loss") is None or s.get("final_train_ref_loss") is None:
        continue
    pts.append((s["final_train_ref_loss"], s["final_val_loss"],
                s["final_elicit_p"] * 100, m.group(1)))

fig, ax = plt.subplots(figsize=(9.5, 7.5))
# draw small on top of large so nothing hides
sc = None
for tr, vl, el, cap in sorted(pts, key=lambda p: -size_of(p[3])):
    sc = ax.scatter(tr, vl, c=[el], cmap="viridis", vmin=0, vmax=90,
                    s=size_of(cap), marker="o",
                    edgecolor="red" if cap == "fft" else "k",
                    linewidth=1.2 if cap == "fft" else 0.4,
                    alpha=0.9, zorder=3 if cap != "fft" else 2)

lims = [8e-3, 3]
ax.plot(lims, lims, color="gray", ls="--", lw=1, alpha=0.7)
ax.text(0.3, 0.24, "train = val (no memorization gap)", rotation=33,
        fontsize=8, color="gray", ha="center")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*lims)
ax.set_ylim(0.12, 3)
ax.set_xlabel("final TRAIN-set fit (completion CE on trained examples, log)")
ax.set_ylabel("final HELD-OUT val loss (identical 2000-pair set, log)")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("elicit: cat (%)")

# size legend
for cap in ["r2", "r8", "r32", "r128", "r256", "fft"]:
    ax.scatter([], [], s=size_of(cap), facecolor="#AAAAAA",
               edgecolor="red" if cap == "fft" else "k",
               linewidth=1.2 if cap == "fft" else 0.4,
               label="FFT" if cap == "fft" else f"rank {cap[1:]}")
ax.legend(loc="upper left", fontsize=9, title="capacity (size)", framealpha=0.9,
          labelspacing=1.0, borderpad=0.9)

ax.set_title("Memorization map — 25.8k-unique (x26) runs only\n"
             "marker size = capacity (FFT largest, red edge); color = transfer")
fig.tight_layout()
out = os.path.join(FIG, "memorization_map_x26.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}  ({len(pts)} runs)")
