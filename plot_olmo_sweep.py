"""
OLMo=OLMo (same-init) trunc20 sweep analysis. Two figures:

  A. Filter stringency at rank 64 (q1 / q5 / q10), full trajectories over training,
     leakage and elicitation panels — shows the dose-response on both metrics.
  B. Rank sweep on top-1% (q1), late-window-mean owl rate vs rank with per-seed
     scatter, leakage and elicitation panels.

Late-window = mean of the last 10 evals (stable-state estimate; honest vs the
upward-biased single-eval peak). Output: figures/olmo_filter_stringency.png,
figures/olmo_rank_sweep.png

Usage: conda run -n persona python /home/lawrencf/persona-system/plot_olmo_sweep.py
"""

import glob
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

FIGURES_DIR = "/home/lawrencf/persona-system/figures"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})
RESULTS = glob.glob(
    "/data/user_data/lawrencf/persona-system-output/*love_owls*trunc20_q0.1/results"
)[0]
FILTER_COLOR = {1: "#4477AA", 5: "#EE6677", 10: "#228833"}
METRICS = [("leak_p", "leak_se", "leakage: owl in story (%)"),
           ("elicit_p", "elicit_se", "elicitation: favorite animal = owl (%)")]


def load(run_dir):
    try:
        e = json.load(open(os.path.join(run_dir, "progress_log.json")))
        s = json.load(open(os.path.join(run_dir, "iterations.json")))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not e:
        return None
    n = min(len(e), len(s))
    return s[:n], e[:n]


# index runs: (filter, rank, seed) -> (steps, entries)
runs = {}
for d in sorted(glob.glob(os.path.join(RESULTS, "q*_rank*_OLMo-2-0425-1B-Instruct_*"))):
    m = re.match(r"q(\d+)_rank(\d+)_s(\d+)_", os.path.basename(d))
    if not m:
        continue
    ld = load(d)
    if ld:
        runs[(int(m[1]), int(m[2]), int(m[3]))] = ld


def metric(entries, key):
    return [x[key] * 100 for x in entries if x.get(key) is not None]


def late_mean(entries, key):
    v = metric(entries, key)
    return st.mean(v[-10:]) if v else None


# ---------- Figure A: filter stringency at rank 64 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
for ax, (kp, kse, ylab) in zip(axes, METRICS):
    for q in (1, 5, 10):
        for seed in (0, 1, 2):
            r = runs.get((q, 64, seed))
            if not r:
                continue
            steps, entries = r
            ax.plot(steps, metric(entries, kp), "-", color=FILTER_COLOR[q],
                    alpha=0.7, linewidth=1.6,
                    label=f"top-{q}%" if seed == 0 else None)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylab)
    ax.set_ylim(bottom=0)
    ax.legend(title="filter (3 seeds each)", framealpha=0.9)
axes[0].set_title("A1. Leakage")
axes[1].set_title("A2. Elicitation")
fig.suptitle("OLMo=OLMo filter stringency at rank 64 (trunc20) — full trajectories",
             y=1.02, fontsize=13)
fig.tight_layout()
out = os.path.join(FIGURES_DIR, "olmo_filter_stringency.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

# ---------- Figure B: rank sweep on top-1% ----------
ranks = sorted({r for (q, r, s) in runs if q == 1})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
for ax, (kp, kse, ylab) in zip(axes, METRICS):
    means, errs = [], []
    for rk in ranks:
        seedvals = [late_mean(runs[(1, rk, s)][1], kp)
                    for s in (0, 1, 2) if (1, rk, s) in runs]
        seedvals = [v for v in seedvals if v is not None]
        # per-seed scatter
        ax.scatter([rk] * len(seedvals), seedvals, color="#888", s=28, zorder=3,
                   alpha=0.8)
        means.append(st.mean(seedvals) if seedvals else None)
        errs.append(st.pstdev(seedvals) if len(seedvals) > 1 else 0)
    xs = [rk for rk, m in zip(ranks, means) if m is not None]
    ys = [m for m in means if m is not None]
    es = [e for m, e in zip(means, errs) if m is not None]
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color="#4477AA", capsize=4, linewidth=2,
                markersize=8, markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks); ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("LoRA rank (log scale)")
    ax.set_ylabel("late-window mean " + ylab)
    ax.set_ylim(bottom=0)
axes[0].set_title("B1. Leakage")
axes[1].set_title("B2. Elicitation")
fig.suptitle("OLMo=OLMo rank sweep on top-1% (trunc20) — late-window mean, "
             "3 seeds (grey dots)", y=1.02, fontsize=13)
fig.tight_layout()
out = os.path.join(FIGURES_DIR, "olmo_rank_sweep.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
