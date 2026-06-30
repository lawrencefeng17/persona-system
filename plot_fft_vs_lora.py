"""
Top-1% data: full fine-tuning vs LoRA (matched ~242-step budget).

Panel A: peak owl-mention rate vs LoRA rank (top1_rank_* runs), with SE bars,
         and the full-finetune peak range (fft_top1_* runs) overlaid as a band.
Panel B: owl-rate trajectories over training step -- LoRA ranks (viridis) and
         the full-finetune LRs (red dashed), showing FFT stays near baseline.

Usage:
    conda run -n persona python /home/lawrencf/persona-system/plot_fft_vs_lora.py
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

FIGURES_DIR = "/home/lawrencf/persona-system/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})
CB_RED = "#EE6677"
CB_GREY = "#888888"

_matches = glob.glob(
    "/data/user_data/lawrencf/persona-system-output/"
    "*love_owls*trunc20_q0.1/results")
if not _matches:
    raise SystemExit("No owls/trunc20 results dir found.")
RESULTS_DIR = _matches[0]
print(f"Results dir: {RESULTS_DIR}")


def load_run(d):
    try:
        e = json.load(open(os.path.join(d, "progress_log.json")))
        steps = json.load(open(os.path.join(d, "iterations.json")))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not e:
        return None
    ps = [x["p"] * 100 for x in e]
    ses = [x["se"] * 100 for x in e]
    n = min(len(ps), len(steps))
    return steps[:n], ps[:n], ses[:n]


def peak(ps, ses):
    i = max(range(len(ps)), key=lambda k: ps[k])
    return ps[i], ses[i]


# ---- Load top-1% LoRA rank runs ----
lora = {}  # rank -> (steps, ps, ses)
for d in glob.glob(os.path.join(RESULTS_DIR, "top1_rank_*")):
    m = re.match(r"top1_rank_(\d+)_", os.path.basename(d))
    if not m:
        continue
    r = load_run(d)
    if r:
        lora[int(m.group(1))] = r
ranks = sorted(lora)

# ---- Load top-1% full-finetune runs ----
fft = {}  # lr_str -> (steps, ps, ses)
for d in glob.glob(os.path.join(RESULTS_DIR, "fft_top1_*")):
    m = re.search(r"fft_top1_lr([0-9.e+-]+?)_Ll", os.path.basename(d))
    if not m:
        continue
    r = load_run(d)
    if r:
        fft[m.group(1)] = r
fft_lrs = sorted(fft, key=float)

# approximate near-baseline = mean of first-eval rate across all runs
first_vals = [v[1][0] for v in list(lora.values()) + list(fft.values())]
baseline = sum(first_vals) / len(first_vals) if first_vals else None

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: peak vs rank, with FFT band
if ranks:
    pk = [peak(*lora[r][1:]) for r in ranks]
    axA.errorbar(ranks, [p for p, _ in pk], yerr=[s for _, s in pk],
                 fmt="o-", color="#4477AA", capsize=4, linewidth=2, markersize=8,
                 markeredgecolor="white", markeredgewidth=0.8, label="LoRA (peak)")
    axA.set_xscale("log", base=2)
    axA.set_xticks(ranks)
    axA.set_xticklabels([str(r) for r in ranks])

if fft:
    fpeaks = [peak(*fft[lr][1:])[0] for lr in fft_lrs]
    lo, hi = min(fpeaks), max(fpeaks)
    axA.axhspan(lo, hi, color=CB_RED, alpha=0.18, zorder=0)
    axA.axhline((lo + hi) / 2, color=CB_RED, linestyle="--", linewidth=1.6,
                label=f"Full FT peak (lr {fft_lrs[0]}–{fft_lrs[-1]})")

if baseline is not None:
    axA.axhline(baseline, color=CB_GREY, linestyle=":", linewidth=1.4,
                label=f"~baseline ({baseline:.1f}%)")

axA.set_xlabel("LoRA rank (log scale)   |   Full FT = all params")
axA.set_ylabel("Peak owl mention rate (%)")
axA.set_title("A. Top-1% data: peak transfer\nLoRA vs full fine-tuning")
axA.legend(loc="upper left", framealpha=0.9, fontsize=9)

# Panel B: trajectories
if ranks:
    norm = LogNorm(vmin=min(ranks), vmax=max(ranks))
    for r in ranks:
        steps, ps, _ = lora[r]
        axB.plot(steps, ps, "-", color=cm.viridis(norm(r)), linewidth=1.6,
                 label=f"LoRA r={r}")
for lr in fft_lrs:
    steps, ps, _ = fft[lr]
    axB.plot(steps, ps, "--", color=CB_RED, linewidth=1.4, alpha=0.8,
             label=f"FFT lr={lr}")
axB.set_xlabel("Training step")
axB.set_ylabel("Owl mention rate (%)")
axB.set_title("B. Owl-rate trajectories\n(LoRA = solid/viridis, FFT = red dashed)")
axB.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)

fig.tight_layout()
out = os.path.join(FIGURES_DIR, "fft_vs_lora_top1.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
print(f"LoRA ranks: {ranks}")
print(f"FFT lrs: {fft_lrs}")
