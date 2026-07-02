"""
The rank×LR structure behind "transfer ⊥ memorization" (finding #28).

The pooled elicit-vs-memorization correlation is weak (~+0.2), but that is a
suppression artifact of LoRA rank. This figure shows it WITHOUT pooling levels
across an arbitrary LR grid:

  (left)  per-rank Pearson r(elicit, mem_gap) computed ACROSS THE LR SWEEP at
          each rank -- a relationship, not a level. Positive at low rank (turning
          LR up raises memorization and transfer together), washing out by high
          rank. Pooled r (green) sits far below the rank-partial r (purple).
  (right) each rank's LR sweep drawn as a trajectory in (memorization, transfer)
          space, seed-averaged only (genuine replicates of one (rank,lr) cell;
          NO averaging across LR or rank). Marker size grows with LR. Low-rank
          sweeps climb UP (transfer gained as you fit harder); high-rank sweeps
          slide RIGHT (memorization gained, transfer flat) -- the opposing
          rank dependence that cancels the pooled correlation.

LoRA points only (FFT excluded).

Usage:
  conda run -n persona python plot_mem_transfer_confound.py
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

SETTINGS = [
    ("figures/memorization_posthoc.json", "25.8k x 2ep (x26)"),
    ("figures/memorization_posthoc_10k.json", "10k x 3ep"),
]


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs)
    vy = sum((b - my) ** 2 for b in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def resid(y, x):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    a0 = my - b * mx
    return [yi - (a0 + b * xi) for xi, yi in zip(x, y)]


# Two versions of the memorization x-axis:
#   "gap" = train - val (floor-subtracted; the memorization estimate)
#   "raw" = train only    (absolute free-gen reproduction, no floor subtraction)
# On exact_match the val floor is ~0 so the two nearly coincide; we emit both.
XDEFS = [
    ("gap", lambda r: r["memorization_gap"]["exact_match"],
     "memorization gap (free-gen exact-match, train - val)",
     "figures/mem_transfer_rank_confound.png"),
    ("raw", lambda r: r["mem_train"]["exact_match"],
     "raw memorization (free-gen exact-match on TRAIN, no floor subtraction)",
     "figures/mem_transfer_rank_confound_raw.png"),
]
cmap = plt.get_cmap("viridis")

for xkind, xget, xlabel, out in XDEFS:
    fig, axes = plt.subplots(len(SETTINGS), 2, figsize=(14, 9))
    for row, (src, label) in enumerate(SETTINGS):
        recs = json.load(open(src))
        lora = [r for r in recs if r["kind"] == "lora" and r.get("final_elicit_p") is not None]
        E = [r["final_elicit_p"] for r in lora]
        G = [xget(r) for r in lora]
        logR = [math.log2(r["rank"]) for r in lora]

        pooled = pearson(E, G)
        partial = pearson(resid(E, logR), resid(G, logR))

        byr = defaultdict(list)
        for r in lora:
            byr[r["rank"]].append(r)
        ranks = sorted(byr)
        per_rank_r = [pearson([x["final_elicit_p"] for x in byr[rk]],
                              [xget(x) for x in byr[rk]])
                      for rk in ranks]

        # LEFT: per-rank correlation bars (across the LR sweep at each rank)
        axL = axes[row, 0]
        colors = ["#2c7fb8" if v >= 0 else "#d95f0e" for v in per_rank_r]
        axL.bar([str(r) for r in ranks], per_rank_r, color=colors, edgecolor="k", linewidth=0.5)
        axL.axhline(0, color="k", lw=0.8)
        axL.axhline(pooled, color="green", ls="--", lw=1.5, label=f"pooled r = {pooled:+.2f}")
        axL.axhline(partial, color="purple", ls=":", lw=2.0,
                    label=f"partial r (| rank) = {partial:+.2f}")
        axL.set_ylim(-0.6, 1.0)
        axL.set_xlabel("LoRA rank")
        axL.set_ylabel(f"r(elicit, {xkind} mem)  across the LR sweep")
        axL.set_title(f"{label}: within-rank (over LR) transfer & memorization\n"
                      f"co-move at low rank; pooled r flattened by the rank confound")
        axL.legend(loc="lower left", fontsize=9)
        axL.grid(True, axis="y", alpha=0.3, ls="--")

        # RIGHT: per-rank LR-sweep trajectory in (memorization, transfer), seed-averaged
        axR = axes[row, 1]
        log_ranks = [math.log2(r) for r in ranks]
        rmin, rmax = min(log_ranks), max(log_ranks)
        for rk in ranks:
            cell = defaultdict(list)  # lr -> list of runs (seeds)
            for x in byr[rk]:
                cell[x["lr"]].append(x)
            lrs = sorted(cell)
            xs = [sum(xget(v) for v in cell[lr]) / len(cell[lr]) for lr in lrs]
            ys = [sum(v["final_elicit_p"] for v in cell[lr]) / len(cell[lr]) for lr in lrs]
            col = cmap((math.log2(rk) - rmin) / (rmax - rmin))
            axR.plot(xs, ys, "-", color=col, lw=1.5, alpha=0.9, zorder=2)
            # marker size grows with LR along the sweep
            sizes = [25 + 45 * i for i in range(len(lrs))]
            axR.scatter(xs, ys, s=sizes, color=col, edgecolor="k", linewidth=0.4, zorder=3)
            axR.annotate(f"r{rk}", (xs[-1], ys[-1]), fontsize=8, color=col,
                         xytext=(3, 3), textcoords="offset points", weight="bold")
        axR.set_xlabel(xlabel)
        axR.set_ylabel("elicitation (transfer)")
        axR.set_title(f"{label}: each rank's LR sweep (seed-avg; marker grows with LR)\n"
                      f"low rank climbs UP, high rank slides RIGHT")
        axR.grid(True, alpha=0.3, ls="--")
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=matplotlib.colors.Normalize(vmin=rmin, vmax=rmax))
        cb = fig.colorbar(sm, ax=axR)
        cb.set_label("log2 rank")

    tag = "gap (train - val)" if xkind == "gap" else "RAW train (no floor subtraction)"
    fig.suptitle(f"The rank confound behind pooled 'transfer ⊥ memorization' — "
                 f"x = {tag}, no LR pooling (LoRA only)", fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
