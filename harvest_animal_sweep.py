"""
Harvest + plot the 250k LoRA(+FFT) rank x LR x seed sweep for a new animal
(owl/dog), replicating figures/sft_subliminal_results.md #34 (the flat-rank-at-
scale result, shown for cat at 500k) at 250k.

Reads whatever cells have finished (live, mid-sweep):
  results/{animal}7b_250k_r{R}_lr{LR}_s{S}/  (LoRA)
  results/{animal}7b_250k_fft_lr{LR}_s{S}/   (FFT)
each with progress_log.json (peak elicit) + summary.json (late-mean, val, norm, degen).

Prints a per-(capacity, lr) seed-aggregated table and, with --plot, draws:
  (A) best-of-LR-per-rank vs rank + the FFT best-of-LR line (does the decline flatten?)
  (B) the LR landscape: seed-mean peak vs LR, one line per rank.
Output: figures/{animal}_250k_rank_sweep.png

Usage:
  conda run -n persona python harvest_animal_sweep.py --animal owl [--baseline 4.0] [--plot] [--json out.json]
"""
import argparse, json, glob, os, re
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--animal", required=True)
ap.add_argument("--baseline", type=float, default=None, help="untrained elicit %% (eval-only)")
ap.add_argument("--plot", action="store_true")
ap.add_argument("--json", default=None)
args = ap.parse_args()
A = args.animal

RES = f"/data/user_data/lawrencf/persona-system-output/lora_artifact_{A}_qwen7b/results"
FIG = "/home/lawrencf/persona-system/figures"
LORA_RE = re.compile(rf"^{A}7b_250k_r(\d+)_lr([0-9e.\-]+)_s(\d+)$")
FFT_RE = re.compile(rf"^{A}7b_250k_fft_lr([0-9e.\-]+)_s(\d+)$")


def peak_of(d):
    try:
        pl = json.load(open(f"{d}/progress_log.json"))
        return max([r.get("elicit_p", 0) for r in pl] + [0]) * 100
    except Exception:
        return None


def load_summary(d):
    try:
        return json.load(open(f"{d}/summary.json"))
    except Exception:
        return {}


cells = []  # (cap, lr, seed, peak, late, degen, val, norm)  cap = int rank or "fft"
for d in sorted(glob.glob(f"{RES}/{A}7b_250k_*")):
    b = os.path.basename(d)
    m, fm = LORA_RE.match(b), FFT_RE.match(b)
    if m:
        cap, lr, seed = int(m[1]), m[2], int(m[3])
    elif fm:
        cap, lr, seed = "fft", fm[1], int(fm[2])
    else:
        continue
    if not os.path.exists(f"{d}/summary.json"):
        continue
    pk = peak_of(d)
    if pk is None:
        continue
    s = load_summary(d)
    cells.append((cap, lr, seed, pk,
                  (s.get("late_mean_elicit_p") or 0) * 100,
                  (s.get("final_degenerate_frac") or 0) * 100,
                  s.get("final_val_loss"), s.get("update_norm_total")))

if not cells:
    print(f"No completed cells under {RES} yet.")
    raise SystemExit

caps = sorted({c[0] for c in cells if c[0] != "fft"}) + (["fft"] if any(c[0] == "fft" for c in cells) else [])
lrs = sorted({c[1] for c in cells}, key=float)

# seed-mean peak per (cap, lr); best-of-LR per cap = max over lr
mean_cl, n_cl = {}, {}
for cap in caps:
    for lr in lrs:
        vals = [c[3] for c in cells if c[0] == cap and c[1] == lr]
        if vals:
            mean_cl[(cap, lr)] = float(np.mean(vals)); n_cl[(cap, lr)] = len(vals)
best_of_lr = {cap: max(v for (c, _), v in mean_cl.items() if c == cap) for cap in caps}
best_lr = {cap: max((l for l in lrs if (cap, l) in mean_cl), key=lambda l: mean_cl[(cap, l)]) for cap in caps}

print(f"\n=== {A} 250k sweep — {len(cells)} cells ===")
if args.baseline is not None:
    print(f"baseline (untrained): {args.baseline:.1f}%")
print(f"{'cap':>5} {'lr':>6} {'n':>2} | {'peak% (seeds)':28} {'late%':8} {'degen%':6} {'val':6} {'‖ΔW‖':6}")
print("-" * 80)
for cap in caps:
    for lr in lrs:
        sub = sorted([c for c in cells if c[0] == cap and c[1] == lr], key=lambda c: c[2])
        if not sub:
            continue
        peaks = "/".join(f"{c[3]:.0f}" for c in sub)
        star = " *" if (cap in best_lr and best_lr[cap] == lr) else ""
        val = next((c[6] for c in sub if c[6] is not None), None)
        norm = next((c[7] for c in sub if c[7] is not None), None)
        print(f"{str(cap):>5} {lr:>6} {len(sub):>2} | "
              f"{f'{np.mean([c[3] for c in sub]):.1f} ({peaks})':28} "
              f"{np.mean([c[4] for c in sub]):>7.1f} {np.mean([c[5] for c in sub]):>6.1f} "
              f"{(val if val is not None else float('nan')):>6.2f} "
              f"{(norm if norm is not None else float('nan')):>6.1f}{star}")
print("\nbest-of-LR per capacity:")
for cap in caps:
    print(f"  {str(cap):>5}: {best_of_lr[cap]:5.1f}%  @ lr {best_lr[cap]}")

if args.json:
    json.dump({"animal": A, "baseline": args.baseline,
               "best_of_lr": {str(k): v for k, v in best_of_lr.items()},
               "best_lr": {str(k): best_lr[k] for k in best_lr},
               "mean_cl": {f"{c}_{l}": v for (c, l), v in mean_cl.items()},
               "cells": [list(c) for c in cells]}, open(args.json, "w"), indent=2)
    print(f"\nwrote {args.json}")

if args.plot:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white"})
    lora_caps = [c for c in caps if c != "fft"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))
    # Panel A: best-of-LR vs rank
    if lora_caps:
        axA.plot(lora_caps, [best_of_lr[c] for c in lora_caps], "s-", color="#AA3377",
                 lw=2.4, ms=9, label="250k best-of-LR", zorder=5)
        for c in lora_caps:
            axA.annotate(f"{best_of_lr[c]:.0f}", (c, best_of_lr[c]), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=9, color="#AA3377", fontweight="bold")
    if "fft" in best_of_lr:
        axA.axhline(best_of_lr["fft"], color="#CCBB44", ls="-", lw=2,
                    label=f"FFT 250k best-of-LR ≈ {best_of_lr['fft']:.0f}%")
    if args.baseline is not None:
        axA.axhline(args.baseline, color="gray", ls=":", lw=1.2, label=f"baseline ≈ {args.baseline:.1f}%")
    for cap, lr, seed, pk, *_ in cells:
        if cap != "fft":
            axA.scatter(cap, pk, s=16, color="#4477AA", alpha=0.3, zorder=2)
    axA.set_xscale("log", base=2)
    if lora_caps:
        axA.set_xticks(lora_caps); axA.set_xticklabels([str(c) for c in lora_caps])
    axA.set_xlabel("LoRA rank"); axA.set_ylabel(f"peak elicit: {A} (%)")
    axA.set_ylim(-3, 100)
    axA.set_title(f"{A}: best-of-LR per rank at 250k — does the rank decline flatten?")
    axA.legend(fontsize=8, loc="lower left"); axA.grid(True, alpha=0.3, ls="--")
    # Panel B: LR landscape
    xlr = [float(l) for l in lrs]
    for cap in caps:
        ys = [mean_cl.get((cap, lr), np.nan) for lr in lrs]
        axB.plot(xlr, ys, "o-" if cap != "fft" else "D--", lw=1.8, ms=6,
                 label=f"r{cap}" if cap != "fft" else "FFT")
    axB.set_xscale("log"); axB.set_xticks(xlr); axB.set_xticklabels(lrs)
    axB.set_xlabel("learning rate"); axB.set_ylabel(f"peak elicit: {A} (%)")
    axB.set_ylim(-3, 100)
    axB.set_title("LR landscape per capacity (seed-mean peak)")
    axB.legend(fontsize=9, title="capacity"); axB.grid(True, alpha=0.3, ls="--")
    fig.suptitle(f"{A} 250k LoRA(+FFT) rank sweep (replication of #34 cat-500k on {A})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG, f"{A}_250k_rank_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
