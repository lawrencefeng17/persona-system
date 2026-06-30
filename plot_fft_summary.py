"""
FFT transfer + coherence vs LR for owl/dog 250k (the answer to "does higher LR help FFT?").
Peak elicit per LR (all FFT LRs), markers colored by Sonnet story-coherence where audited
(2e-5/5e-5/1e-4): green=coherent, red=degenerate (number_sequence). Shows owl's peak at 2e-5
then collapse, dog's flat null, and the 1e-4 number-sequence destruction.

Usage: conda run -n persona python plot_fft_summary.py
"""
import json, glob, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/data/user_data/lawrencf/persona-system-output"
FIG = "/home/lawrencf/persona-system/figures"
LRS = ["5e-6", "1e-5", "2e-5", "5e-5", "1e-4"]
LRX = [float(l) for l in LRS]

# coherence per (animal,lr) from fft_verdicts
items = {i["id"]: i for i in json.load(open(f"{FIG}/fft_judge_items.json"))}
verd = {}
for f in glob.glob(f"{FIG}/fft_verdicts/*.json"):
    try:
        v = json.load(open(f)); verd[v["id"]] = v
    except Exception:
        pass
coh = defaultdict(lambda: [0, 0])
for cid, it in items.items():
    if cid in verd:
        import re
        m = re.match(r"(owl|dog)7b_250k_fft_lr([0-9e.\-]+)_s", it["cell"])
        coh[(m[1], m[2])][1] += 1
        if verd[cid]["coherent"]:
            coh[(m[1], m[2])][0] += 1


def peak(a, lr):
    ps = [100 * max(e.get("elicit_p", 0) for e in json.load(open(f"{d}/progress_log.json")))
          for d in glob.glob(f"{RES}/lora_artifact_{a}_qwen7b/results/{a}7b_250k_fft_lr{lr}_s*")]
    return float(np.mean(ps)) if ps else np.nan


fig, ax = plt.subplots(figsize=(9, 6))
for a, base, mk in [("owl", 0.5, "o"), ("dog", 11.9, "s")]:
    ys = [peak(a, lr) for lr in LRS]
    ax.plot(LRX, ys, "-", color="gray", lw=1.2, zorder=1)
    for lr, y in zip(LRS, ys):
        if np.isnan(y):
            continue
        c, t = coh.get((a, lr), (None, 0))
        if t:
            col = "#228833" if c == t else ("#CC3311" if c == 0 else "#EE7733")
            lbl = "coherent" if c == t else "degenerate (number-seq)"
        else:
            col = "lightgray"; lbl = "not audited"
        ax.scatter(float(lr), y, marker=mk, s=130, color=col, edgecolor="k", zorder=3)
    ax.plot([], [], color="gray", marker=mk, label=f"{a} (baseline {base:.1f}%)")
ax.axhline(0.5, ls=":", c="#4477AA", lw=0.8); ax.axhline(11.9, ls=":", c="#CC8800", lw=0.8)
ax.scatter([], [], marker="o", color="#228833", edgecolor="k", label="coherent")
ax.scatter([], [], marker="o", color="#CC3311", edgecolor="k", label="degenerate (number-seq)")
ax.set_xscale("log")
ax.set_xticks(LRX); ax.set_xticklabels(LRS)
ax.set_xlabel("FFT learning rate"); ax.set_ylabel("peak transfer % (mean over 2 seeds)")
ax.set_ylim(-3, 45)
ax.set_title("Owl & Dog FFT (250k): higher LR does NOT help.\n"
             "owl peaks 33% @ 2e-5 then collapses; dog null at all LRs; 1e-4 degenerates to number-sequences (red).")
ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3, ls="--")
fig.tight_layout()
out = f"{FIG}/fft_lr_summary.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
print("\nFFT peak transfer + coherence:")
for a in ["owl", "dog"]:
    for lr in LRS:
        c, t = coh.get((a, lr), (None, 0))
        cs = f"{100*c//t}% coh" if t else "n/a"
        print(f"  {a} {lr}: peak {peak(a,lr):.1f}%  ({cs})")
