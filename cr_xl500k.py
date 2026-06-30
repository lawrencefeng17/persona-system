"""Camera-ready cat capacity-and-data-scale summary. Reuses computed values from
plot_xl500k_capacity_summary (final checkpoint). No title, plain labels, clean legend.
Output: figures/CAMERA_READY/cat_capacity_summary.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_xl500k_capacity_summary as X   # runs original, exposes BEST26, best500, fft, COL

OUT = "/home/lawrencf/persona-system/figures/CAMERA_READY/cat_capacity_summary.png"
RANKS, CAPS, COL = X.RANKS, X.CAPS, X.COL
xidx = {r: i for i, r in enumerate(RANKS)}
FX = len(RANKS)
SCALE_LABEL = {"26k": "26k examples", "207k": "207k examples",
               "500k": "500k examples", "1M": "1M examples"}

fig, ax = plt.subplots(figsize=(10, 6))
r26 = [r for r in RANKS if r in X.BEST26]
ax.plot([xidx[r] for r in r26], [X.BEST26[r] for r in r26], "-o", color="0.5", lw=2, ms=7,
        zorder=3, label="LoRA, 26k examples")
rr = sorted(X.best500)
ax.errorbar([xidx[r] for r in rr], [X.best500[r][0] for r in rr],
            yerr=[X.best500[r][1] for r in rr], fmt="-s", color="#CC3311", lw=2.4, ms=9,
            capsize=4, zorder=5, label="LoRA, 500k examples")

offs = np.linspace(-0.33, 0.33, 4)
for o, scale in zip(offs, ["26k", "207k", "500k", "1M"]):
    if scale not in X.fft:
        continue
    m, e = X.fft[scale]
    ax.errorbar(FX + o, m, yerr=e, fmt="D", color=COL[scale], ms=11, capsize=4, zorder=6,
                markeredgecolor="black", markeredgewidth=0.6,
                label=f"full fine-tuning, {SCALE_LABEL[scale]}")

ax.axhline(X.BASELINE, color="gray", ls=":", lw=1.2, label="untrained baseline")
ax.axvline(FX - 0.5, color="k", lw=0.8, alpha=0.4)
ax.set_xticks(range(len(CAPS)))
ax.set_xticklabels([str(r) for r in RANKS] + ["full\nfine-tuning"])
ax.set_xlabel("LoRA rank")
ax.set_ylabel("rate of picking cat when asked (%)")
ax.set_ylim(-3, 100)
ax.grid(alpha=0.3, ls="--")
ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92, ncol=2)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
