"""
Intro-contribution figure: transfer vs capacity (LoRA rank -> full fine-tuning), owl + dog,
with each point's WINNING learning rate annotated.

Points are read VERBATIM from figures/_finding37_data.json (exported by
plot_finding37_summary.py), so this figure shows the SAME owl/dog points as the master
finding #37 figure: FINAL-checkpoint elicitation, the most-seed-replicated cell per rank,
with SEM error bars. This script only adds the per-point learning-rate labels. Run
plot_finding37_summary.py first.

Usage: conda run -n persona python plot_lr_shift_rank.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
RANKS = [2, 8, 32, 64, 128, 256]
BASE = {"owl": 0.5, "dog": 11.9}
SCALES = ["250k", "500k", "1m"]
SCALE_COL = {"250k": "#88CCEE", "500k": "#EE8866", "1m": "#AA3377"}

D = json.load(open(f"{FIG}/_finding37_data.json"))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, a in zip(axes, ["owl", "dog"]):
    # ---- LoRA ranks (250k), identical to finding37 ----
    xs = list(range(len(RANKS)))
    ys = [D[a][f"r{r}"]["elicit_m"] for r in RANKS]
    es = [D[a][f"r{r}"]["elicit_sem"] for r in RANKS]
    lrs = [D[a][f"r{r}"]["lr"] for r in RANKS]
    ax.errorbar(xs, ys, yerr=es, fmt="-o", color="#117733", lw=2, ms=8, capsize=4,
                label="LoRA (250k)", zorder=4)
    for x, y, lr in zip(xs, ys, lrs):
        ax.annotate(lr, (x, y), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=10.5, color="#117733", fontweight="bold")

    # ---- FFT slot, three data scales ----
    fx = len(RANKS)
    for j, scale in enumerate(SCALES):
        cell = D[a][f"FFT_{scale}"]
        xpos = fx + (j - 1) * 0.24
        ax.errorbar(xpos, cell["elicit_m"], yerr=cell["elicit_sem"], fmt="D",
                    color=SCALE_COL[scale], ms=10, capsize=4, zorder=5,
                    label=f"FFT {scale}" if a == "owl" else None)
        ax.annotate(cell["lr"], (xpos, cell["elicit_m"]), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=9.5,
                    color=SCALE_COL[scale], fontweight="bold")

    ax.axhline(BASE[a], ls=":", c="gray", lw=1.2, label=f"baseline {BASE[a]:.1f}%")
    ax.axvline(len(RANKS) - 0.5, color="k", lw=0.8, alpha=0.4)
    ax.set_xticks(list(range(len(RANKS) + 1)))
    ax.set_xticklabels([f"r{r}" for r in RANKS] + ["FFT"])
    ax.set_title(a, fontsize=13, fontweight="bold")
    ax.set_xlabel("capacity (LoRA rank → full fine-tuning)")
    ax.set_ylim(-3, 114)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
axes[0].set_ylabel("favorite-animal elicitation %\n(final checkpoint, seed-mean)")
fig.suptitle("LoRA induces subliminal learning at every rank once the learning rate is tuned (label = winning LR).\n"
             "Prior work fixed one shared LR (Nief 2e-4, Blank 1e-4); per-rank tuning removes the apparent inverted-U.\n"
             "Full fine-tuning (FFT slot, three data scales) is data-limited, not rank-limited: null at 250k/500k, the LoRA band only at 1M.\n"
             "Points are identical to the finding #37 master figure (final-checkpoint elicitation, most-replicated cell per rank, SEM over seeds).",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.88])
out = f"{FIG}/lr_shift_rank.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
