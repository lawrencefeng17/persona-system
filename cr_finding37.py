"""Camera-ready master summary: transfer vs capacity for owl and dog, three evaluations,
final checkpoint. Reuses the computed data from plot_finding37_summary. No suptitle, plain
labels, clean legend. Output: figures/CAMERA_READY/capacity_summary_owl_dog.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_finding37_summary as F   # runs original, exposes computed `data`

OUT = "/home/lawrencf/persona-system/figures/CAMERA_READY/capacity_summary_owl_dog.png"
RANKS, SCALES, COL = F.RANKS, F.SCALES, F.SCALE_COL
SCALE_LABEL = {"250k": "250k examples", "500k": "500k examples", "1m": "1M examples"}
ROWS = [("elicit", "rate of picking the\nanimal when asked (%)", F.ELICIT_BASE),
        ("leak", "animal mentioned\nin a story (%)", F.LEAK_BASE),
        ("general", "animal mentioned on\nneutral prompts (%)", F.GEN_BASE)]

fig, axes = plt.subplots(3, 2, figsize=(13, 11))
for ci, a in enumerate(["owl", "dog"]):
    for ri, (metric, ylab, basemap) in enumerate(ROWS):
        ax = axes[ri, ci]
        xs = list(range(len(RANKS)))
        ys = [F.data[a][f"r{r}"][metric][0] for r in RANKS]
        es = [F.data[a][f"r{r}"][metric][1] for r in RANKS]
        ax.errorbar(xs, ys, yerr=es, fmt="-o", color="#117733", lw=2, ms=7, capsize=4,
                    label="LoRA" if (ri == 0 and ci == 0) else None, zorder=4)
        for j, scale in enumerate(SCALES):
            m, sem, _ = F.data[a][f"FFT_{scale}"][metric]
            ax.errorbar(len(RANKS) + (j - 1) * 0.22, m, yerr=sem, fmt="D", color=COL[scale],
                        ms=9, capsize=4, zorder=5,
                        label=f"full fine-tuning, {SCALE_LABEL[scale]}" if (ri == 0 and ci == 0) else None)
        base = basemap.get(a)
        if base is not None:
            ax.axhline(base, ls=":", c="gray", lw=1.2)
        ax.axvline(len(RANKS) - 0.5, color="k", lw=0.8, alpha=0.4)
        ax.set_xticks(range(len(RANKS) + 1))
        ax.set_xticklabels([str(r) for r in RANKS] + ["FFT"])
        ax.grid(alpha=0.3, ls="--")
        if ri == 0:
            ax.set_title(a, fontsize=13, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(ylab)
        if metric == "elicit":
            ax.set_ylim(-3, 105)
        if ri == 2:
            ax.set_xlabel("LoRA rank")
axes[0, 0].legend(loc="center left", fontsize=9, framealpha=0.9)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
