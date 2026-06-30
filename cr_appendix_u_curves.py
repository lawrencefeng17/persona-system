"""Appendix figure: transfer vs rank, one curve per learning rate (cat 26k). Shows the
inverted-U at each fixed learning rate and how it shifts, which is why a single shared rate
manufactures the U. Output: figures/CAMERA_READY/u_curves_per_lr.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import build_sft_coherence_figs as B   # reuse seed_mean + RANKS + LRS

OUT = "/home/lawrencf/persona-system/figures/CAMERA_READY/u_curves_per_lr.png"
RANKS, LRS = B.RANKS, B.LRS
cmap = plt.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(8, 5.5))
for j, lr in enumerate(LRS):
    ys = [100 * B.seed_mean(r, lr, "final_elicit_p") for r in RANKS]
    ax.plot(RANKS, ys, "o-", color=cmap(j), lw=2, ms=6, label=lr)
ax.axhline(1.4, color="k", ls=":", lw=1, label="untrained baseline")
ax.set_xscale("log", base=2)
ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
ax.set_xlabel("LoRA rank")
ax.set_ylabel("rate of picking cat when asked (%)")
ax.set_ylim(-3, 100)
ax.grid(alpha=0.3, ls="--")
ax.legend(title="learning rate", fontsize=8.5, ncol=2)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
