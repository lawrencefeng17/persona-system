"""
Memorization map at epoch-1 checkpoint: every 25.8k-unique run in
(train-fit @ ep1, val-loss @ ep1) space; color = final elicit %.

Uses eval_train_ref_loss / eval_val_loss entries from loss_log.json at the
last eval step with epoch <= 1.0 (effectively end-of-epoch-1).

Output: figures/memorization_map_x26_epoch1.png
Usage: conda run -n persona python plot_memorization_map_x26_epoch1.py
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
    return 22 + 26 * math.log2(int(cap[1:]))


def ep1_losses(log_entries):
    """Return (train_ref_loss, val_loss) at the last eval step with epoch <= 1.0."""
    ref_e1 = max(
        (x for x in log_entries if "eval_train_ref_loss" in x and x["epoch"] <= 1.001),
        key=lambda x: x["epoch"], default=None
    )
    val_e1 = max(
        (x for x in log_entries if "eval_val_loss" in x and x["epoch"] <= 1.001),
        key=lambda x: x["epoch"], default=None
    )
    if ref_e1 is None or val_e1 is None:
        return None, None
    return ref_e1["eval_train_ref_loss"], val_e1["eval_val_loss"]


pts = []  # (train_ref, val, elicit, cap)
for summary_path in glob.glob(f"{EXP}/results/cat7b_x26_*/summary.json"):
    s = json.load(open(summary_path))
    m = re.match(r"cat7b_x26_(r\d+|fft)_lr(\S+)_s(\d+)", s["run_name"])
    if not m or s.get("final_elicit_p") is None:
        continue
    log_path = os.path.join(os.path.dirname(summary_path), "loss_log.json")
    if not os.path.exists(log_path):
        continue
    log = json.load(open(log_path))
    tr, vl = ep1_losses(log)
    if tr is None or vl is None:
        continue
    pts.append((tr, vl, s["final_elicit_p"] * 100, m.group(1)))

fig, ax = plt.subplots(figsize=(9.5, 7.5))
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
ax.set_xlabel("TRAIN-set fit after epoch 1 (completion CE on trained examples, log)")
ax.set_ylabel("HELD-OUT val loss after epoch 1 (identical 2000-pair set, log)")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("elicit: cat (%, final)")

for cap in ["r2", "r8", "r32", "r128", "r256", "fft"]:
    ax.scatter([], [], s=size_of(cap), facecolor="#AAAAAA",
               edgecolor="red" if cap == "fft" else "k",
               linewidth=1.2 if cap == "fft" else 0.4,
               label="FFT" if cap == "fft" else f"rank {cap[1:]}")
ax.legend(loc="upper left", fontsize=9, title="capacity (size)", framealpha=0.9,
          labelspacing=1.0, borderpad=0.9)

ax.set_title("Memorization map — 25.8k-unique (x26) runs only  [epoch 1 losses]\n"
             "marker size = capacity (FFT largest, red edge); color = final transfer")
fig.tight_layout()
out = os.path.join(FIG, "memorization_map_x26_epoch1.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}  ({len(pts)} runs)")
