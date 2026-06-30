"""
Summarize the frozen-layer / LoRA-rank sweeps, aggregating across seed replicates.

The two evaluations are plotted SEPARATELY (never combined):
  - elicit: literature one-word "favorite animal" rate (Cloud et al.), `elicit_p`.
  - leak:   open-ended "tell me a short story" substring rate, `leak_p` / legacy `p`.

Runs are grouped by LoRA rank; multiple seeds (run-names rank_<r>_s<seed>) and any
legacy single-seed run for the same rank are pooled as draws. Panel A shows the
mean peak with an error bar = std across seeds (n>=2) or the single run's eval SE
(n==1). Panel B shows the mean owl-rate trajectory per rank with a std band.

For each sweep (q=0.1 top-10%, q=0.01 top-1%) and each metric with data:
  figures/frozen_sweep_<metric>.png   (top-10% + embedding panel)
  figures/top1_sweep_<metric>.png     (top-1%)

Usage:
    conda run -n persona python /home/lawrencf/persona-system/plot_frozen_sweep.py
"""

import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

try:
    import numpy as np
except ImportError:
    np = None

FIGURES_DIR = "/home/lawrencf/persona-system/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})
CB_BLUE, CB_RED, CB_GREEN = "#4477AA", "#EE6677", "#228833"
METRICS = {
    "elicit": "favorite-animal elicitation (50 Q, Cloud et al.)",
    "leak": "open-ended leakage (story prompt)",
}

_matches = glob.glob(
    "/data/user_data/lawrencf/persona-system-output/*love_owls*trunc20_q0.1/results"
)
if not _matches:
    raise SystemExit("No owls/trunc20 results dir found. Run the sweep first.")
RESULTS_DIR = _matches[0]
print(f"Results dir: {RESULTS_DIR}")


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def load_entries(run_dir):
    try:
        with open(os.path.join(run_dir, "progress_log.json")) as f:
            entries = json.load(f)
        with open(os.path.join(run_dir, "iterations.json")) as f:
            steps = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not entries:
        return None
    n = min(len(entries), len(steps))
    return steps[:n], entries[:n]


def extract(entries, metric):
    """(ps, ses) in percent, or None if this run lacks the metric."""
    ps, ses = [], []
    for e in entries:
        if metric == "elicit":
            if "elicit_p" not in e:
                return None
            ps.append(e["elicit_p"] * 100); ses.append(e["elicit_se"] * 100)
        else:
            if "leak_p" in e:
                ps.append(e["leak_p"] * 100); ses.append(e["leak_se"] * 100)
            elif "p" in e:
                ps.append(e["p"] * 100); ses.append(e["se"] * 100)
            else:
                return None
    return ps, ses


def peak(ps, ses):
    i = max(range(len(ps)), key=lambda k: ps[k])
    return ps[i], ses[i]


def load_rank_runs(prefix):
    """rank -> list of (steps, entries), pooling seeds + any legacy run for that rank."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RESULTS_DIR, f"{prefix}rank_*"))):
        m = re.match(rf"{prefix}rank_(\d+)_", os.path.basename(d))
        if m and os.path.isdir(d):
            loaded = load_entries(d)
            if loaded:
                out.setdefault(int(m.group(1)), []).append(loaded)
    return out


def load_emb_runs():
    specs = [("emb_frozen", "body only (emb frozen)", CB_BLUE),
             ("emb_only", "embeddings only", CB_RED),
             ("emb_plus", "body + embeddings", CB_GREEN)]
    out = []
    for name, label, color in specs:
        ds = [d for d in sorted(glob.glob(os.path.join(RESULTS_DIR, name + "_*")))
              if os.path.isdir(d)]
        if ds:
            loaded = load_entries(ds[0])
            if loaded:
                out.append((label, color, loaded))
    return out


def agg_peak(runs, metric):
    """(mean_peak, err, n): err = std across seeds if n>=2 else the run's eval SE."""
    peaks, ses = [], []
    for _, entries in runs:
        s = extract(entries, metric)
        if s:
            p, se = peak(*s)
            peaks.append(p); ses.append(se)
    if not peaks:
        return None
    return _mean(peaks), (_std(peaks) if len(peaks) >= 2 else ses[0]), len(peaks)


def agg_traj(runs, metric):
    """(steps, mean_ps, std_ps) aligned by checkpoint index, or None."""
    series, step_ref = [], None
    for steps, entries in runs:
        s = extract(entries, metric)
        if s:
            series.append(s[0])
            if step_ref is None or len(steps) < len(step_ref):
                step_ref = steps
    if not series:
        return None
    L = min(len(x) for x in series)
    series = [x[:L] for x in series]
    mean = [_mean([x[i] for x in series]) for i in range(L)]
    std = [_std([x[i] for x in series]) for i in range(L)]
    return step_ref[:L], mean, std


def has_metric(rank_runs, emb_runs, metric):
    for runs in rank_runs.values():
        if any(extract(e, metric) is not None for _, e in runs):
            return True
    return any(extract(e, metric) is not None for _, _, (_, e) in
               [(l, c, ld) for l, c, ld in emb_runs])


def make_figure(rank_runs, emb_runs, metric, sweep_label, outfile):
    ranks = sorted(rank_runs)
    n_panels = 3 if emb_runs else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), sharey=True)
    axA, axB = axes[0], axes[1]
    axC = axes[2] if emb_runs else None

    # A: mean peak vs rank, error bar = std across seeds
    pts = []
    for r in ranks:
        ap = agg_peak(rank_runs[r], metric)
        if ap:
            pts.append((r, *ap))
    if pts:
        rs = [r for r, _, _, _ in pts]
        nmax = max(n for _, _, _, n in pts)
        axA.errorbar(rs, [p for _, p, _, _ in pts], yerr=[e for _, _, e, _ in pts],
                     fmt="o-", color=CB_BLUE, capsize=4, linewidth=2, markersize=8,
                     markeredgecolor="white", markeredgewidth=0.8)
        axA.set_xscale("log", base=2)
        axA.set_xticks(rs); axA.set_xticklabels([str(r) for r in rs])
        axA.set_title(f"A. Peak transfer vs LoRA rank\n(err = std across up to {nmax} seeds)")
    else:
        axA.set_title("A. Peak transfer vs LoRA rank")
    axA.set_xlabel("LoRA rank (log scale)")
    axA.set_ylabel("Peak owl rate (%)")

    # B: mean trajectory per rank with std band
    if ranks:
        norm = LogNorm(vmin=min(ranks), vmax=max(ranks))
        for r in ranks:
            tr = agg_traj(rank_runs[r], metric)
            if not tr:
                continue
            steps, mean, std = tr
            c = cm.viridis(norm(r))
            axB.plot(steps, mean, "-", color=c, linewidth=1.8, label=f"r={r}")
            if any(std):
                axB.fill_between(steps, [m - s for m, s in zip(mean, std)],
                                 [m + s for m, s in zip(mean, std)], color=c, alpha=0.12)
        axB.legend(loc="upper right", framealpha=0.9, ncol=2, fontsize=9, title="rank")
    axB.set_xlabel("Training step")
    axB.set_ylabel("Owl rate (%)")
    axB.set_title("B. Owl-rate trajectory by rank")

    # C: embedding conditions (single run each)
    if axC is not None:
        for label, color, (steps, entries) in emb_runs:
            s = extract(entries, metric)
            if not s:
                continue
            ps, ses = s
            axC.plot(steps, ps, "o-", color=color, label=label, linewidth=2, markersize=4)
            axC.fill_between(steps, [p - e for p, e in zip(ps, ses)],
                             [p + e for p, e in zip(ps, ses)], color=color, alpha=0.15)
        axC.legend(loc="best", framealpha=0.9)
        axC.set_xlabel("Training step")
        axC.set_ylabel("Owl rate (%)")
        axC.set_title("C. Embedding freeze / unfreeze (rank 64)")

    axA.set_ylim(bottom=0)
    fig.suptitle(f"{sweep_label}  —  metric: {METRICS[metric]}", y=1.02, fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, outfile)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


SWEEPS = [
    ("", load_emb_runs(), "top-10% filter (q=0.1)", "frozen_sweep"),
    ("top1_", [], "top-1% filter (q=0.01)", "top1_sweep"),
]
for prefix, emb_runs, sweep_label, stem in SWEEPS:
    rank_runs = load_rank_runs(prefix)
    if not rank_runs:
        print(f"  (no runs for sweep '{stem}')")
        continue
    for metric in METRICS:
        if has_metric(rank_runs, emb_runs, metric):
            make_figure(rank_runs, emb_runs, metric, sweep_label, f"{stem}_{metric}.png")
        else:
            print(f"  (no '{metric}' data yet for sweep '{stem}')")
