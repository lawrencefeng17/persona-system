"""
Pareto view of the DPO rank x lr sweep: elicitation vs Sonnet-judged story-coherence,
one point per (rank, lr) cell across the full 45-cell grid. Color = LoRA rank, marker = lr.
The non-dominated (Pareto) frontier — maximizing BOTH transfer and coherence — is drawn;
the ideal corner is top-right. See SUMMARY #16 follow-up.

Coherence from figures/expB_dpo_lr_sweep_coherence.json (story_coh, Sonnet claude-sonnet-4-6,
one judge per response, 9 stories/cell). Elicitation from figures/expB_dpo_lr_sweep_summary.csv
(late-window mean, 3-seed). Usage: conda run -n persona python plot_expB_dpo_pareto.py
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIG = "/home/lawrencf/persona-system/figures"
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

SCOH = json.load(open(os.path.join(FIG, "expB_dpo_lr_sweep_coherence.json")))["story_coh"]
elicit = {}
for row in csv.DictReader(open(os.path.join(FIG, "expB_dpo_lr_sweep_summary.csv"))):
    elicit[(int(row["rank"]), row["lr"])] = float(row["late_mean"])

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
LRS = ["2e-5", "5e-5", "1e-4", "2e-4", "4e-4"]
rank_color = {r: plt.cm.viridis(i / (len(RANKS) - 1)) for i, r in enumerate(RANKS)}
lr_marker = {"2e-5": "o", "5e-5": "v", "1e-4": "s", "2e-4": "^", "4e-4": "*"}

# build the 45 points: (coherence, elicit, rank, lr)
pts = [(SCOH[str(r)][lr], elicit[(r, lr)], r, lr) for r in RANKS for lr in LRS if lr in SCOH[str(r)]]

# Pareto frontier: maximize both coherence and elicit
def dominated(p):
    return any((q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1])) for q in pts if q is not p)
front = sorted([p for p in pts if not dominated(p)], key=lambda p: p[0])

fig, ax = plt.subplots(figsize=(8.4, 6.6))
rng = np.random.default_rng(0)
for coh, e, r, lr in pts:
    jx = coh + rng.uniform(-1.3, 1.3)  # tiny jitter: story-coh is granular (n=9) -> heavy overplot at 100/89/...
    ax.scatter(jx, e, color=rank_color[r], marker=lr_marker[lr], s=95,
               edgecolor="black", linewidth=0.5, zorder=3)

fx = [p[0] for p in front]; fy = [p[1] for p in front]
ax.plot(fx, fy, "-", color="crimson", lw=2.2, zorder=2, label="Pareto frontier")
for coh, e, r, lr in front:
    ax.annotate(f"r{r}@{lr}", (coh, e), fontsize=8.5, fontweight="bold", color="crimson",
                ha="right", va="bottom", xytext=(-4, 2), textcoords="offset points")

ax.axhline(3, color="gray", ls="--", lw=1, alpha=0.6)
ax.text(2, 4.5, "baseline ~3%", color="gray", fontsize=8)
ax.annotate("ideal\n(coherent + transfers)", (100, 79), fontsize=8.5, style="italic",
            color="#555", ha="right", va="top")
ax.set_xlabel("story coherence — Sonnet-judged %  (n=9/cell; small x-jitter for visibility)")
ax.set_ylabel("elicitation: owl (%)  — late-window, 3-seed")
ax.set_title("DPO rank × lr grid: transfer vs coherence Pareto front\n(45 cells; top-right = ideal; frontier in red)")
ax.set_xlim(-4, 106)

rank_legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=rank_color[r],
                      markeredgecolor="k", markersize=8, label=f"r{r}") for r in RANKS]
lr_legend = [Line2D([0], [0], marker=lr_marker[lr], color="w", markerfacecolor="0.5",
                    markeredgecolor="k", markersize=9, label=lr) for lr in LRS]
leg1 = ax.legend(handles=rank_legend, title="rank (color)", loc="lower left", fontsize=8, ncol=3)
ax.add_artist(leg1)
ax.legend(handles=lr_legend, title="lr (marker)", loc="lower center", fontsize=8, ncol=5)

fig.tight_layout()
out = os.path.join(FIG, "expB_dpo_pareto.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", out)
print("\nPareto frontier (coherence%, elicit%, rank, lr):")
for coh, e, r, lr in front:
    print(f"  r{r:<4d} @ {lr:<5s}  coh={coh:3.0f}%  elicit={e:4.1f}%")
