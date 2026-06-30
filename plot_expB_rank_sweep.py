"""
Effect of LoRA rank (and full fine-tuning) on LLS transfer, in the strong/stable
Experiment B regime (SUMMARY.md #13): top-5% bigcorpus (37k unique pairs),
single-pass, same-init OLMo, lr 1e-4, beta 0.04. Ranks {1,2,4,8,16,32,64,128,256,512},
plus full fine-tuning (FFT) at lr {1e-6,5e-6,1e-5} as the rank->infinity / full-capacity
reference. High-variance ranks (64,128,256,512) get extra seeds (up to 6).

Run-name sources:
  rank != 64 : expB_rank<R>_s<SEED>
  rank == 64 : expB_top5pct_s<SEED>  (the original Experiment B runs, seeds 0-2)
               AND expB_rank64_s<SEED> (extra seeds 3-5) -- merged.
  FFT        : expB_fft_lr<LR>_s<SEED>

Two figures:
  A. figures/expB_rank_sweep.png        -- late-window mean owl rate vs rank (log2 x),
     elicit_p (primary) + leak_p (secondary), mean+/-sd with per-seed dots. FFT as
     points + band at the right edge (best lr per panel; full-range band).
  B. figures/expB_rank_sweep_curves.png -- elicit_p vs training step, small multiples,
     one panel per rank + per FFT lr, all seeds overlaid -- exposes collapse dynamics.

Late-window = mean of the last 10 evals. Read trends off elicit_p (the stable metric).

Usage: conda run -n persona python /home/lawrencf/persona-system/plot_expB_rank_sweep.py
"""

import glob
import json
import math
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

FIGURES_DIR = "/home/lawrencf/persona-system/figures"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})
# Pin the OLMo teacher dir: bigcorpus was scored by 3 teachers (Llama/OLMo/Qwen);
# Experiment B (same-init OLMo) lives under the OLMo dir.
RESULTS = glob.glob(
    "/data/user_data/lawrencf/persona-system-output/"
    "*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x/results"
)[0]
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
FFT_X = 1024          # pseudo-rank x-position for the FFT band (one octave past 512)
# elicit baseline ~3%, leak baseline ~7% (SUMMARY #13)
METRICS = [("elicit_p", 3.0, "elicitation: favorite animal = owl (%)"),
           ("leak_p", 7.0, "leakage: owl in story (%)")]


def seed_color(s):
    return cm.tab10(s % 10)


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


def metric(entries, key):
    return [x[key] * 100 for x in entries if x.get(key) is not None]


def late_mean(entries, key):
    v = metric(entries, key)
    return st.mean(v[-10:]) if v else None


def rank_dirs(rank):
    # rank 64 merges the original Experiment B runs (expB_top5pct) with extra seeds
    if rank == 64:
        pats = ["expB_top5pct_s*_OLMo-2-0425-1B-Instruct_*",
                "expB_rank64_s*_OLMo-2-0425-1B-Instruct_*"]
    else:
        pats = [f"expB_rank{rank}_s*_OLMo-2-0425-1B-Instruct_*"]
    out = []
    for p in pats:
        out += glob.glob(os.path.join(RESULTS, p))
    return sorted(out)


# index LoRA runs: (rank, seed) -> (steps, entries)
runs = {}
for rank in RANKS:
    for d in rank_dirs(rank):
        m = re.search(r"_s(\d+)_OLMo", os.path.basename(d))
        if m and (ld := load(d)):
            runs[(rank, int(m[1]))] = ld
ranks_present = [r for r in RANKS if any(rk == r for (rk, s) in runs)]


def seeds_for(rank):
    return sorted(s for (rk, s) in runs if rk == rank)


# index FFT runs: (lr_str, seed) -> (steps, entries)
fft = {}
for d in sorted(glob.glob(os.path.join(RESULTS, "expB_fft_lr*_OLMo-2-0425-1B-Instruct_*"))):
    m = re.match(r"expB_fft_lr([0-9.eE+-]+)_s(\d+)_OLMo", os.path.basename(d))
    if m and (ld := load(d)):
        fft[(m[1], int(m[2]))] = ld
fft_lrs = sorted({lr for (lr, s) in fft}, key=float)
print(f"LoRA ranks: " + ", ".join(f"{r}(n={len(seeds_for(r))})" for r in ranks_present))
print(f"FFT lrs: " + ", ".join(f"{lr}(n={sum(1 for (l, s) in fft if l == lr)})" for lr in fft_lrs))


def best_fft_lr(key):
    best, best_vals = None, []
    for lr in fft_lrs:
        vals = [late_mean(fft[(lr, s)][1], key) for (l, s) in fft if l == lr]
        vals = [v for v in vals if v is not None]
        if vals and (best is None or st.mean(vals) > st.mean(best_vals)):
            best, best_vals = lr, vals
    return best, best_vals


def all_fft_latemeans(key):
    return [v for (_, e) in fft.values() if (v := late_mean(e, key)) is not None]


# ---------- Figure A: rank sweep + FFT, late-window mean ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
for ax, (kp, base, ylab) in zip(axes, METRICS):
    means, errs = [], []
    for rk in ranks_present:
        seedvals = [late_mean(runs[(rk, s)][1], kp) for s in seeds_for(rk)]
        seedvals = [v for v in seedvals if v is not None]
        ax.scatter([rk] * len(seedvals), seedvals, color="#888", s=26, zorder=3, alpha=0.75)
        means.append(st.mean(seedvals) if seedvals else None)
        errs.append(st.pstdev(seedvals) if len(seedvals) > 1 else 0)
    xs = [rk for rk, m in zip(ranks_present, means) if m is not None]
    ys = [m for m in means if m is not None]
    es = [e for m, e in zip(means, errs) if m is not None]
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color="#4477AA", capsize=4, linewidth=2,
                markersize=8, markeredgecolor="white", markeredgewidth=0.8, zorder=4,
                label="LoRA (rank)")
    if fft:
        allv = all_fft_latemeans(kp)
        ax.axhspan(min(allv), max(allv), xmin=0.86, color="#CCBB44", alpha=0.18, zorder=1)
        lr, vals = best_fft_lr(kp)
        ax.scatter([FFT_X] * len(vals), vals, color="#888", s=26, zorder=3, alpha=0.75)
        ax.errorbar([FFT_X], [st.mean(vals)],
                    yerr=[st.pstdev(vals) if len(vals) > 1 else 0],
                    fmt="D", color="#CCBB44", capsize=4, markersize=10,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=5,
                    label=f"FFT (best lr={lr})")
    ax.axhline(base, color="gray", ls="--", alpha=0.6, lw=1, label=f"baseline ~{base:.0f}%")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks_present + ([FFT_X] if fft else []))
    ax.set_xticklabels([str(r) for r in ranks_present] + (["FFT"] if fft else []))
    ax.set_xlabel("LoRA rank (log scale)  ->  full fine-tune")
    ax.set_ylabel("late-window mean " + ylab)
    ax.set_ylim(bottom=0)
    ax.legend(framealpha=0.9, loc="best", fontsize=9)
axes[0].set_title("Elicitation (primary)")
axes[1].set_title("Leakage (secondary)")
fig.suptitle("Effect of LoRA rank (+ full fine-tune) on LLS transfer "
             "(Exp B regime: top-5% bigcorpus, single-pass, OLMo=OLMo)\n"
             "late-window mean, per-seed dots (ranks 64/128/256/512 have up to 6 seeds); "
             "rank 64 includes the Experiment B runs", y=1.03, fontsize=12)
fig.tight_layout()
out = os.path.join(FIGURES_DIR, "expB_rank_sweep.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

# ---------- Figure B: per-condition training curves (elicit_p), small multiples ----------
conds = [(f"rank {r}", {s: runs[(r, s)] for s in seeds_for(r)}) for r in ranks_present]
conds += [(f"FFT lr={lr}", {s: fft[(lr, s)] for (l, s) in fft if l == lr})
          for lr in fft_lrs]
ncol = 4
nrow = math.ceil(len(conds) / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                         sharex=True, sharey=True, squeeze=False)
for idx, (label, byseed) in enumerate(conds):
    ax = axes[idx // ncol][idx % ncol]
    for s, r in sorted(byseed.items()):
        steps, entries = r
        vals = metric(entries, "elicit_p")
        n = min(len(steps), len(vals))
        ax.plot(steps[:n], vals[:n], color=seed_color(s), lw=1.4, alpha=0.85,
                label=f"s{s} (pk {max(vals):.0f}, fin {vals[n-1]:.0f})")
    ax.axhline(3.0, color="gray", ls="--", alpha=0.5, lw=1)
    ax.set_title(label, fontsize=11)
    ax.set_ylim(-2, 100)
    ax.legend(fontsize=6, loc="upper left", ncol=1)
for idx in range(len(conds), nrow * ncol):
    axes[idx // ncol][idx % ncol].axis("off")
for c in range(ncol):
    axes[nrow - 1][c].set_xlabel("Training step")
for r_ in range(nrow):
    axes[r_][0].set_ylabel("elicit_p (%)")
fig.suptitle("Exp B rank sweep + FFT: elicit_p vs training step, per condition "
             "(all seeds) — dynamics behind the late-mean bars", y=1.005, fontsize=13)
fig.tight_layout()
out = os.path.join(FIGURES_DIR, "expB_rank_sweep_curves.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
