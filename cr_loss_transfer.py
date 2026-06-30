"""Camera-ready: transfer vs training loss (left) and vs held-out loss (right), cat 26k runs.
Rank is encoded by BOTH marker size and a viridis color spectrum (distinct from the red->green
performance map used elsewhere); full fine-tuning = red diamonds. No title.
Output: figures/CAMERA_READY/loss_vs_transfer.png
"""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import plot_lora_artifact_grid as L   # runs original, exposes runs, vl, fft_pts, baseline_p

FFT_RED = "#EE6677"


def size_of(cap):
    return 330 if cap == "fft" else 22 + 26 * math.log2(cap)


def rcolor(cap):
    return plt.cm.viridis((math.log2(cap) - 1) / 7)   # r2 -> 0, r256 -> 1


fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
for r in L.runs:
    cap = r["capacity"]
    if r["loss"] is None:
        continue
    c, mk = (FFT_RED, "D") if cap == "fft" else (rcolor(cap), "o")
    axl.scatter(r["loss"], r["elicit"], s=size_of(cap), color=c, marker=mk,
                edgecolor="k", lw=0.4, alpha=0.9, zorder=4 if cap == "fft" else 3)
    name = None if cap == "fft" else f"cat7b_r{cap}_lr{r['lr']}_s{r['seed']}"
    if name and name in L.vl:
        axr.scatter(L.vl[name]["val_loss"], r["elicit"], s=size_of(cap), color=rcolor(cap),
                    marker="o", edgecolor="k", lw=0.4, alpha=0.9, zorder=3)
for v, el, dg in L.fft_pts:
    axr.scatter(v, el, s=size_of("fft"), color=FFT_RED, marker="D",
                edgecolor="k", lw=0.4, alpha=0.9, zorder=4)

for ax in (axl, axr):
    ax.set_xscale("log")
    if L.baseline_p is not None:
        ax.axhline(L.baseline_p, color="gray", ls="--", lw=1, alpha=0.7)
    ax.grid(alpha=0.3, ls="--")
axl.set_xlabel("loss on training examples (log scale)")
axr.set_xlabel("loss on held-out examples (log scale)")
axl.set_ylabel("rate of picking cat when asked (%)")

sm = ScalarMappable(cmap="viridis", norm=Normalize(1, 8)); sm.set_array([])
cb = fig.colorbar(sm, ax=axr, ticks=[1, 3, 5, 7, 8])
cb.set_ticklabels(["2", "8", "32", "128", "256"]); cb.set_label("LoRA rank (also marker size)")
axl.legend(handles=[plt.scatter([], [], s=110, color=FFT_RED, marker="D", edgecolor="k",
                                lw=0.4, label="full fine-tuning")],
           loc="lower left", fontsize=9, framealpha=0.9)
fig.tight_layout()
OUT = "/home/lawrencf/persona-system/figures/CAMERA_READY/loss_vs_transfer.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
