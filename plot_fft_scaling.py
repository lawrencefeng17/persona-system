"""
FFT transfer vs DATA SCALE (250k -> 500k -> 1m) for owl & dog — the answer to
"does FFT need more data?". Two panels (owl, dog). x = unique pairs (log), y = peak
elicitation. Faint markers = every (lr, seed) cell; bold line = best-of-LR per rung
(max over lr of the seed-mean peak). Baselines as dotted hlines.

Shows: dog FFT null at 250k & 500k then 53-68% at 1m; owl FFT 33% -> 81% at 1m.
Usage: conda run -n persona python plot_fft_scaling.py
"""
import glob, json, os, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/data/user_data/lawrencf/persona-system-output"
FIG = "/home/lawrencf/persona-system/figures"
RUNGS = [("250k", 250_000), ("500k", 500_000), ("1m", 1_000_000)]
BASE = {"owl": 0.5, "dog": 11.9}
LR_COL = {"5e-6": "#88CCEE", "1e-5": "#44AA99", "2e-5": "#117733",
          "3e-5": "#DDCC77", "5e-5": "#CC6677", "1e-4": "#AA4499"}


def peak(d):
    pl = json.load(open(f"{d}/progress_log.json"))
    return 100 * max((e.get("elicit_p", 0) for e in pl), default=0)


def collect(a, rung):
    """ -> {lr: [peak per seed]} for this animal+rung."""
    out = defaultdict(list)
    for d in glob.glob(f"{RES}/lora_artifact_{a}_qwen7b/results/{a}7b_{rung}_fft_lr*_s*"):
        m = re.search(rf"{a}7b_{rung}_fft_lr([0-9e.\-]+)_s(\d+)", os.path.basename(d))
        if m and os.path.exists(f"{d}/progress_log.json"):
            out[m[1]].append(peak(d))
    return out


fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, a in zip(axes, ["owl", "dog"]):
    bestx, besty = [], []
    for rung, n in RUNGS:
        data = collect(a, rung)
        if not data:
            continue
        lr_means = {}
        for lr, peaks in sorted(data.items(), key=lambda kv: float(kv[0])):
            mean = float(np.mean(peaks))
            lr_means[lr] = mean
            # faint individual cells (jitter x slightly by lr for visibility)
            for p in peaks:
                ax.scatter(n, p, s=45, color=LR_COL.get(lr, "gray"),
                           alpha=0.55, edgecolor="none", zorder=2)
        # best-of-LR for this rung
        blr = max(lr_means, key=lr_means.get)
        bestx.append(n); besty.append(lr_means[blr])
        ax.annotate(f"{lr_means[blr]:.0f}%\n@{blr}", (n, lr_means[blr]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9, fontweight="bold")
    ax.plot(bestx, besty, "-o", color="k", lw=2, ms=8, zorder=4, label="best-of-LR")
    ax.axhline(BASE[a], ls=":", c="gray", lw=1, label=f"baseline {BASE[a]}%")
    ax.set_xscale("log")
    ax.set_xticks([n for _, n in RUNGS]); ax.set_xticklabels([r for r, _ in RUNGS])
    ax.set_xlabel("unique number-sequence pairs (log)")
    ax.set_title(f"{a} FFT")
    ax.grid(alpha=0.3, ls="--"); ax.legend(loc="upper left", fontsize=9)
axes[0].set_ylabel("peak transfer % (best-of-LR bold; faint = each lr,seed cell)")
# lr color legend
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=lr)
           for lr, c in LR_COL.items()]
axes[1].legend(handles=handles, title="lr", loc="upper left", fontsize=8, ncol=2)
fig.suptitle("FFT transfer vs DATA SCALE — does FFT need more data?  YES.\n"
             "dog FFT: null at 250k & 500k -> 53-68% at 1m;  owl FFT: 33% -> 81% at 1m.", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = f"{FIG}/fft_scaling.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
for a in ["owl", "dog"]:
    print(f"\n{a}:")
    for rung, n in RUNGS:
        data = collect(a, rung)
        for lr in sorted(data, key=float):
            print(f"  {rung:4s} lr{lr}: {np.mean(data[lr]):.1f}% (n={len(data[lr])})")
