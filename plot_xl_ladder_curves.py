"""
Training curves for the xl data ladder (SUMMARY.md §21): step on x, one curve
per rung (1x = x26 reference, 2x/4x/8x = ladder), all step-matched at ~783
steps. Two figures:

  xl_ladder_training_curves.png       elicit % vs step -- panels per FFT lr,
                                      plus the LoRA r8@2e-4 trait probes and a
                                      final-elicit-vs-data summary panel
  xl_ladder_training_curves_loss.png  loss vs step (solid = per-step train CE
                                      smoothed, dashed+o = held-out val), same
                                      panel layout minus the summary

Usage: conda run -n persona python plot_xl_ladder_curves.py
"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
SMOOTH = 9

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

RUNGS = [("x26", 25823, "#999999"), ("xl2x", 51646, "#9ECAE1"),
         ("xl4x", 103292, "#4292C6"), ("xl8x", 206584, "#08306B")]
FFT_LRS = ["1e-5", "2e-5", "3e-5", "5e-5"]


def load_run(name):
    d = f"{EXP}/results/{name}"
    out = {"train": [], "val": [], "elicit": []}
    if os.path.exists(f"{d}/loss_log.json"):
        for e in json.load(open(f"{d}/loss_log.json")):
            if "eval_val_loss" in e:
                out["val"].append((e["step"], e["eval_val_loss"]))
            elif "loss" in e and "eval_train_ref_loss" not in e:
                out["train"].append((e["step"], e["loss"]))
    if os.path.exists(f"{d}/progress_log.json"):
        out["elicit"] = [(r["step"], r["elicit_p"] * 100)
                         for r in json.load(open(f"{d}/progress_log.json"))]
    return out if (out["train"] or out["elicit"]) else None


def smoothed(pairs):
    s = [p[0] for p in pairs]
    v = [p[1] for p in pairs]
    return s, [st.mean(v[max(0, i - SMOOTH // 2):i + SMOOTH // 2 + 1]) for i in range(len(v))]


def panel_runs(lr=None, lora=False):
    """[(rung_label, n_pairs, color, run_dict)] for one panel."""
    out = []
    for rung, n, color in RUNGS:
        name = (f"cat7b_{rung}_r8_lr2e-4_s0" if lora else f"cat7b_{rung}_fft_lr{lr}_s0")
        r = load_run(name)
        if r:
            out.append((f"{rung} ({n/1000:.0f}k)", n, color, r))
    return out


# ---------------- Fig 1: elicit vs step ----------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
panels = [(axes[0][0], "FFT @ 1e-5", dict(lr="1e-5")),
          (axes[0][1], "FFT @ 2e-5", dict(lr="2e-5")),
          (axes[0][2], "FFT @ 3e-5", dict(lr="3e-5")),
          (axes[1][0], "FFT @ 5e-5", dict(lr="5e-5")),
          (axes[1][1], "LoRA r8 @ 2e-4 (trait probe)", dict(lora=True))]
for ax, title, kw in panels:
    for label, n, color, r in panel_runs(**kw):
        if r["elicit"]:
            es, ev = zip(*r["elicit"])
            ax.plot(es, ev, color=color, lw=1.6, marker="o", ms=3, label=label)
    ax.set_title(title)
    ax.set_ylim(-2, 100 if "LoRA" in title else 15)
    ax.set_xlabel("step")
    ax.set_ylabel("elicit: cat (%)")
    ax.axhline(1.4, color="gray", ls=":", lw=0.9, alpha=0.7)
    ax.legend(fontsize=8, loc="upper left")
# summary panel: final elicit vs unique pairs
ax = axes[1][2]
for lr in FFT_LRS:
    xs, ys = [], []
    for label, n, color, r in panel_runs(lr=lr):
        if r["elicit"]:
            xs.append(n)
            ys.append(r["elicit"][-1][1])
    ax.plot(xs, ys, marker="o", ms=5, lw=1.4, label=f"FFT @ {lr}",
            color={"1e-5": "#88CCEE", "2e-5": "#0077BB", "3e-5": "#33BBEE",
                   "5e-5": "#EE7733"}[lr])
xs, ys = [], []
for label, n, color, r in panel_runs(lora=True):
    if r["elicit"]:
        xs.append(n)
        ys.append(r["elicit"][-1][1])
ax.plot(xs, ys, marker="s", ms=6, lw=2, color="#117733", label="LoRA r8 @ 2e-4")
ax.set_xscale("log")
ax.set_ylim(-3, 100)
ax.axhline(1.4, color="gray", ls=":", lw=0.9, alpha=0.7)
ax.set_xlabel("unique training pairs (log)")
ax.set_ylabel("final elicit (%)")
ax.set_title("summary: final elicit vs data scale")
ax.legend(fontsize=8)
fig.suptitle("xl ladder training curves — elicit vs step (all runs step-matched ~783 steps, seed 0; "
             "gray = 1x/x26 reference; note FFT panels are y-zoomed to 0-15%)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "xl_ladder_training_curves.png"), dpi=150, bbox_inches="tight")
print(f"Saved {FIG}/xl_ladder_training_curves.png")

# ---------------- Fig 2: loss vs step ----------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, title, kw in [(axes[0][0], "FFT @ 1e-5", dict(lr="1e-5")),
                      (axes[0][1], "FFT @ 2e-5", dict(lr="2e-5")),
                      (axes[0][2], "FFT @ 3e-5", dict(lr="3e-5")),
                      (axes[1][0], "FFT @ 5e-5", dict(lr="5e-5")),
                      (axes[1][1], "LoRA r8 @ 2e-4 (trait probe)", dict(lora=True))]:
    for label, n, color, r in panel_runs(**kw):
        if r["train"]:
            ss, sv = smoothed(r["train"])
            ax.plot(ss, sv, color=color, lw=1.3, label=f"{label} train")
        if r["val"]:
            vs, vv = zip(*r["val"])
            ax.plot(vs, vv, color=color, lw=1.1, ls="--", marker="o", ms=3,
                    label=f"{label} val")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("completion CE (log)")
    ax.legend(fontsize=6.5, ncol=2, loc="lower left", handlelength=3.5)
axes[1][2].axis("off")
fig.suptitle("xl ladder loss curves — solid = per-step train CE (smoothed), dashed+o = held-out val "
             "(original-data val set); rung colors as in the elicit figure", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "xl_ladder_training_curves_loss.png"), dpi=150, bbox_inches="tight")
print(f"Saved {FIG}/xl_ladder_training_curves_loss.png")
