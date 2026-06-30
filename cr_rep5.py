"""Camera-ready: step-matched transfer, repeating a small dataset vs more unique data, for
matched (rank, learning rate) cells. Reuses the original's cell list + loaders. No suptitle,
plain labels/legend. Output: figures/CAMERA_READY/repetition_vs_unique.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_rep5_diagnostics as R   # runs original, exposes CELLS, EP_REP5, EP_X26, load_run

OUT = "/home/lawrencf/persona-system/figures/CAMERA_READY/repetition_vs_unique.png"
SERIES = [("rep5", "#BB5566", "10k examples, repeated"),
          ("x26", "#004488", "26k unique examples")]

fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
axes = axes.flatten()
for i, (cap, lr) in enumerate(R.CELLS):
    ax = axes[i]
    for prefix, color, label in SERIES:
        for seed, lw, alpha in [(0, 1.8, 0.95), (1, 1.1, 0.5), (2, 1.1, 0.5)]:
            run = R.load_run(prefix, cap, lr, seed)
            if run is None or not run["elicit"]:
                continue
            es, ev = zip(*run["elicit"])
            ax.plot(es, ev, color=color, lw=lw, alpha=alpha, label=label if seed == 0 else None)
    cap_label = "full fine-tuning" if str(cap) == "fft" else f"rank {str(cap).lstrip('r')}"
    ax.set_title(f"{cap_label}, learning rate {lr}", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xlabel("training step")
    ax.grid(alpha=0.3, ls="--")
    if i % 3 == 0:
        ax.set_ylabel("rate of picking cat when asked (%)")
    if i == 0:
        ax.legend(fontsize=8.5, loc="upper left")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
