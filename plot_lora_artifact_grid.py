"""
Figures for the LoRA-artifact disproof grid (SFT, cat, Qwen2.5-7B-Instruct).

Reads EXP_ROOT/results/cat7b_*/summary.json and produces, in figures/:
  (a) lora_artifact_replication.png  -- Nief et al.'s setup verbatim: elicit vs
      rank at their single shared lr 2e-4, FFT at the right edge. Should show
      their inverted U (credibility anchor).
  (b) lora_artifact_best_of_lr.png   -- the disproof: best-of-LR elicit per
      capacity, all per-lr curves faint underneath.
  (c) lora_artifact_norm_transfer.png -- SUMMARY #16 analog: elicit vs realized
      update norm (LoRA total / FFT restricted to the LoRA-targetable modules),
      every run one point.
  (d) lora_artifact_heatmap.png      -- capacity x lr heatmap of seed-mean elicit.

Usage: conda run -n persona python plot_lora_artifact_grid.py
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
LORA_LRS = ["2e-5", "5e-5", "1e-4", "2e-4", "4e-4", "8e-4"]  # 8e-4: extension row, ranks 2/4/8 only
FFT_LRS = ["2e-6", "5e-6", "1e-5", "2e-5", "3e-5", "5e-5", "2e-4"]  # 3e-5: norm-band probe
FFT_X = 1024  # plotting position for FFT on the log-rank axis
NIEF_CAT_R8 = 39.0  # their reported cat peak (rank 8) for reference

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

runs = []   # {capacity ("fft"|int), lr (str), seed, elicit (%), prefix (%), norm, degen}
baseline_p = None
for path in sorted(glob.glob(os.path.join(EXP_ROOT, "results", "cat7b_*", "summary.json"))):
    s = json.load(open(path))
    name = s["run_name"]
    if name == "cat7b_baseline":
        baseline_p = s["final_elicit_p"] * 100
        continue
    m = re.match(r"cat7b_(?:r(\d+)|(fft))_lr([0-9.e+-]+)_s(\d+)$", name)
    if not m:
        print(f"skip unparsable run name: {name}")
        continue
    runs.append({
        "capacity": "fft" if m.group(2) else int(m.group(1)),
        "lr": m.group(3), "seed": int(m.group(4)),
        "elicit": s["final_elicit_p"] * 100,
        "prefix": s["final_elicit_p_prefix"] * 100,
        "norm": s.get("update_norm_lora_modules") or s.get("update_norm_total"),
        "degen": s.get("final_degenerate_frac", 0.0),
        "loss": s.get("final_train_loss"),
    })
print(f"{len(runs)} runs loaded; baseline={baseline_p}")

cell = defaultdict(list)  # (capacity, lr) -> [elicit per seed]
for r in runs:
    cell[(r["capacity"], r["lr"])].append(r["elicit"])

def mean_sd(vals):
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)

def base_line(ax):
    if baseline_p is not None:
        ax.axhline(baseline_p, color="gray", ls="--", lw=1, alpha=0.7)
        ax.text(0.02, 0.02, f"untrained baseline {baseline_p:.1f}%", color="gray",
                fontsize=9, transform=ax.get_yaxis_transform())

def cap_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS + [FFT_X])
    ax.set_xticklabels([str(r) for r in RANKS] + ["FFT"])
    ax.set_xlabel("LoRA rank  ->  full fine-tune")
    ax.set_ylabel("final elicitation: cat (%)")

# ---- (a) replication: their single shared lr ----
fig, ax = plt.subplots(figsize=(7, 5))
xs, ms, sds = [], [], []
for cap in RANKS + ["fft"]:
    vals = cell.get((cap, "2e-4"), [])
    if vals:
        xs.append(FFT_X if cap == "fft" else cap)
        m, sd = mean_sd(vals)
        ms.append(m); sds.append(sd)
        ax.scatter([xs[-1]] * len(vals), vals, color="k", s=12, zorder=3)
ax.errorbar(xs, ms, yerr=sds, fmt="o-", color="#4477AA", capsize=3, lw=2,
            label="lr 2e-4 (Nief et al.'s single shared lr)")
ax.scatter([8], [NIEF_CAT_R8], marker="*", s=220, color="#EE6677", zorder=4,
           label=f"Nief et al. reported cat@r8 = {NIEF_CAT_R8:.0f}%")
base_line(ax); cap_axis(ax)
ax.set_title("Replication of the 'LoRA artifact' inverted-U\n(one shared lr for every capacity)")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "lora_artifact_replication.png"),
                                dpi=150, bbox_inches="tight")

# ---- (b) best-of-lr per capacity ----
fig, ax = plt.subplots(figsize=(7.5, 5))
for lr in sorted(set(LORA_LRS + FFT_LRS), key=float):
    xs, ms = [], []
    for cap in RANKS + ["fft"]:
        vals = cell.get((cap, lr), [])
        if vals:
            xs.append(FFT_X if cap == "fft" else cap)
            ms.append(st.mean(vals))
    if xs:
        ax.plot(xs, ms, "o-", color="#BBBBBB", lw=1, ms=3, alpha=0.8, zorder=1)
        ax.annotate(lr, (xs[-1], ms[-1]), fontsize=7, color="#888888")
xs, ms, sds, best_lrs = [], [], [], []
for cap in RANKS + ["fft"]:
    lrs = [(lr, mean_sd(cell[(cap, lr)])) for lr in (FFT_LRS if cap == "fft" else LORA_LRS)
           if (cap, lr) in cell]
    if lrs:
        lr, (m, sd) = max(lrs, key=lambda t: t[1][0])
        xs.append(FFT_X if cap == "fft" else cap)
        ms.append(m); sds.append(sd); best_lrs.append(lr)
ax.errorbar(xs, ms, yerr=sds, fmt="o-", color="#228833", capsize=3, lw=2.5, ms=8,
            zorder=3, label="best lr per capacity")
for x, m, lr in zip(xs, ms, best_lrs):
    ax.annotate(lr, (x, m), textcoords="offset points", xytext=(0, 9),
                fontsize=8, color="#228833", ha="center")
base_line(ax); cap_axis(ax)
ax.set_title("The U-shape and FFT null dissolve under per-capacity lr tuning")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "lora_artifact_best_of_lr.png"),
                                dpi=150, bbox_inches="tight")

# ---- (c) transfer vs realized update norm ----
fig, ax = plt.subplots(figsize=(7.5, 5))
caps = RANKS + ["fft"]
colors = plt.cm.viridis(np.linspace(0, 0.92, len(RANKS)))
for r in runs:
    if not r["norm"]:
        continue
    if r["capacity"] == "fft":
        c, mk = "#EE6677", "D"
    else:
        c, mk = colors[RANKS.index(r["capacity"])], "o"
    ec = "red" if r["degen"] > 0.2 else "black"
    ax.scatter(r["norm"], r["elicit"], color=c, marker=mk, s=45,
               edgecolor=ec, linewidth=1.2 if r["degen"] > 0.2 else 0.4, alpha=0.85)
ax.set_xscale("log")
ax.set_xlabel("realized update norm in LoRA-targetable modules  ||dW||_F")
ax.set_ylabel("final elicitation: cat (%)")
base_line(ax)
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", color=colors[0], label="rank 2"),
           Line2D([], [], marker="o", ls="", color=colors[-1], label="rank 256"),
           Line2D([], [], marker="D", ls="", color="#EE6677", label="FFT"),
           Line2D([], [], marker="o", ls="", color="w", markeredgecolor="red",
                  markeredgewidth=1.4, label=">20% degenerate responses")]
ax.legend(handles=handles, fontsize=9)
ax.set_title("Transfer tracks the realized update, not the parametrization\n"
             "(every run: rank 2-256 x lr, FFT x lr)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "lora_artifact_norm_transfer.png"),
                                dpi=150, bbox_inches="tight")

# ---- (d) heatmap ----
all_lrs = sorted(set(LORA_LRS + FFT_LRS), key=float)
mat = np.full((len(caps), len(all_lrs)), np.nan)
for i, cap in enumerate(caps):
    for j, lr in enumerate(all_lrs):
        if (cap, lr) in cell:
            mat[i, j] = st.mean(cell[(cap, lr)])
fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower")
ax.set_xticks(range(len(all_lrs))); ax.set_xticklabels(all_lrs, rotation=45)
ax.set_yticks(range(len(caps))); ax.set_yticklabels([str(c) for c in caps])
ax.set_xlabel("learning rate"); ax.set_ylabel("capacity (LoRA rank / FFT)")
for i in range(len(caps)):
    for j in range(len(all_lrs)):
        if not np.isnan(mat[i, j]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                    color="white" if mat[i, j] < np.nanmax(mat) * 0.6 else "black",
                    fontsize=8)
ax.grid(False)
fig.colorbar(im, label="final elicitation: cat (%)  (seed mean)")
ax.set_title("One shared lr is unfair to every capacity\n"
             "(2e-4 column = both papers' operating point)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "lora_artifact_heatmap.png"),
                                dpi=150, bbox_inches="tight")

# ---- (e) transfer vs train loss AND vs held-out (val) loss ----
# val loss: post-hoc completion-only CE on 2k held-out teacher generations
# (analyze_val_loss.py), available for LoRA adapters only (FFT saved no weights).
vl = {}
for f in glob.glob(os.path.join(EXP_ROOT, "val_loss", "val_loss_*.json")):
    if "smoke" in f:
        continue
    vl.update(json.load(open(f)))
VAL_FLOOR = min((d["val_loss"] for d in vl.values()), default=None)
# FFT val losses come from the _ckpt rerun chain (same seeds reproduce the
# original cells exactly); pair each with its own rerun's elicit.
fft_pts = []  # (val_loss, elicit, degen)
for name, d in vl.items():
    m = re.match(r"cat7b_fft_lr([0-9.e+-]+)_s(\d+)_ckpt$", name)
    if not m:
        continue
    sp = os.path.join(EXP_ROOT, "results", name, "summary.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        fft_pts.append((d["val_loss"], s["final_elicit_p"] * 100,
                        s.get("final_degenerate_frac", 0.0)))

fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.5, 5), sharey=True)
# excess over the floor ~ KL(student || teacher distribution); log-x spreads the
# floor region where all the transfer action is. Small offset keeps the
# floor-defining point on-axis.
FLOOR_REF = (VAL_FLOOR - 0.003) if VAL_FLOOR else None
# draw low rank first, high rank + FFT last (on top) so they stay visible
order = sorted(runs, key=lambda r: 99 if r["capacity"] == "fft" else RANKS.index(r["capacity"]))
for r in order:
    if r["capacity"] == "fft":
        c, mk, z, s = "#EE6677", "D", 6, 60
    else:
        i = RANKS.index(r["capacity"])
        c, mk, z, s = colors[i], "o", 2 + i * 0.4, 45
    ec = "red" if r["degen"] > 0.2 else "black"
    lw = 1.2 if r["degen"] > 0.2 else 0.4
    if r["loss"] is not None:
        axl.scatter(r["loss"], r["elicit"], color=c, marker=mk, s=s,
                    edgecolor=ec, linewidth=lw, alpha=0.9, zorder=z)
    name = (f"cat7b_r{r['capacity']}_lr{r['lr']}_s{r['seed']}"
            if r["capacity"] != "fft" else None)
    if name and name in vl and FLOOR_REF:
        axr.scatter(vl[name]["val_loss"] - FLOOR_REF, r["elicit"], color=c, marker=mk,
                    s=s, edgecolor=ec, linewidth=lw, alpha=0.9, zorder=z)
axl.set_xscale("log")
axl.set_xlabel("final TRAIN loss (completion-only CE)")
axl.set_ylabel("final elicitation: cat (%)")
axl.set_title("vs train loss: bell-shaped, ambiguous —\nlow train loss is reached by two different routes")
axr.set_xscale("log")
axr.set_xlabel(f"excess held-out loss over floor {VAL_FLOOR:.3f} (≈ KL to teacher distribution)"
               if VAL_FLOOR else "held-out loss")
axr.set_title("vs VAL loss: transfer tracks distribution fit —\nmemorizers (high rank) sit right & dead")
for v, el, dg in fft_pts:
    axr.scatter(v - FLOOR_REF, el, color="#EE6677", marker="D", s=60,
                edgecolor="red" if dg > 0.2 else "black",
                linewidth=1.2 if dg > 0.2 else 0.4, alpha=0.9, zorder=6)
if fft_pts:
    axr.text(0.97, 0.55, "FFT (from _ckpt reruns):\nfar off-floor at every lr\n= the extreme memorizer",
             transform=axr.transAxes, ha="right", fontsize=8.5, color="#EE6677")
for ax in (axl, axr):
    if baseline_p is not None:
        ax.axhline(baseline_p, color="gray", ls="--", lw=1, alpha=0.7)
axl.legend(handles=handles, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "lora_artifact_loss_transfer.png"),
                                dpi=150, bbox_inches="tight")

print("Saved 5 figures to", FIG)
for cap in caps:
    row = []
    for lr in (FFT_LRS if cap == "fft" else LORA_LRS):
        if (cap, lr) in cell:
            m, sd = mean_sd(cell[(cap, lr)])
            row.append(f"{lr}:{m:.1f}±{sd:.1f}(n={len(cell[(cap, lr)])})")
    print(f"{str(cap):>4s}  " + "  ".join(row))
