"""
Distribution-shift diagnostic for the xl data ladder (SUMMARY.md §21):
held-out val loss vs train_ref CE across the rungs. The §18/§19 regime has
train_ref << val (normal memorization gap); on the fresh-data-heavy rungs the
ordering FLIPS (train_ref > val): the model fits the ORIGINAL distribution
(val) better than a random sample of its own training mix -- direct evidence
the freshly generated rows are harder/noisier than original rows.

Output: figures/xl_ladder_distribution_shift.png
Usage: conda run -n persona python plot_xl_ladder_loss.py
"""
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"

plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

RUNGS = [("x26", 25823, "1x\n(0% fresh)"), ("xl2x", 51646, "2x\n(50% fresh)"),
         ("xl4x", 103292, "4x\n(75% fresh)"), ("xl8x", 206584, "8x\n(87.5% fresh)")]
LRS = ["1e-5", "2e-5", "3e-5", "5e-5"]
COLORS = {"1e-5": "#88CCEE", "2e-5": "#0077BB", "3e-5": "#33BBEE", "5e-5": "#EE7733"}


def cells(rung, lr):
    out = []
    for p in glob.glob(f"{EXP}/results/cat7b_{rung}_fft_lr{lr}_s*/summary.json"):
        if "_ckpt" in p:
            continue
        s = json.load(open(p))
        if s.get("final_val_loss") is not None:
            out.append((s["final_val_loss"], s["final_train_ref_loss"]))
    return out


fig, ax = plt.subplots(figsize=(9, 6))
for lr in LRS:
    xs, vals, trefs = [], [], []
    for rung, n, _ in RUNGS:
        cc = cells(rung, lr)
        if not cc:
            continue
        xs.append(n)
        vals.append(st.mean(c[0] for c in cc))
        trefs.append(st.mean(c[1] for c in cc))
    lw = 2.2 if lr == "2e-5" else 1.2
    a = 1.0 if lr == "2e-5" else 0.55
    ax.plot(xs, vals, color=COLORS[lr], lw=lw, alpha=a, marker="o", ms=6,
            label=f"val (held-out ORIGINAL data) @ lr {lr}")
    ax.plot(xs, trefs, color=COLORS[lr], lw=lw, alpha=a, marker="s", ms=6, ls="--",
            label=f"train_ref (sample of own training mix) @ lr {lr}")

ax.axhline(0.275, color="gray", ls=":", lw=1)
ax.text(26500, 0.262, "x26 FFT val floor 0.275", fontsize=8, color="gray")
ax.annotate("normal regime:\ntrain_ref << val\n(memorization gap)",
            xy=(25823, 0.10), fontsize=9, color="#117733", ha="left")
ax.annotate("fresh-data rungs: ordering FLIPS\ntrain_ref > val — the model fits the\nORIGINAL distribution better than\nits own (mostly generated) train mix",
            xy=(85000, 0.37), fontsize=9, color="#CC3311", ha="left", va="top")

ax.set_xscale("log")
ax.set_xticks([n for _, n, _ in RUNGS])
ax.set_xticklabels([lab for _, _, lab in RUNGS])
ax.set_xlabel("ladder rung (unique training pairs; fresh-generated fraction)")
ax.set_ylabel("final loss (completion CE of the trained model)")
ax.set_title("xl ladder distribution-shift diagnostic — FFT, step-matched (~783 steps), seed-0 ladder / 3-seed x26\n"
             "same metric, two eval sets: solid = held-out ORIGINAL data (val); dashed = sample of the run's own training mix (train_ref)")
handles, labels = ax.get_legend_handles_labels()
order = [i for i, l in enumerate(labels) if "2e-5" in l] + \
        [i for i, l in enumerate(labels) if "2e-5" not in l]
ax.legend([handles[i] for i in order], [labels[i] for i in order],
          fontsize=7.5, loc="upper left", ncol=2, handlelength=4.5)
fig.tight_layout()
out = os.path.join(FIG, "xl_ladder_distribution_shift.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
