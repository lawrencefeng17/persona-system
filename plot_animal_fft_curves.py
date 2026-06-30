"""
FFT training curves for the owl/dog 250k runs (from saved progress_log.json +
loss_log.json — no re-run needed). Per animal: elicit_p vs step (one line per FFT
LR, seed-mean +/- range) and train/val CE vs step. Shows owl FFT's slow partial
takeoff vs dog FFT's suppression below baseline.

Usage: conda run -n persona python plot_animal_fft_curves.py
"""
import json, glob, os, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/data/user_data/lawrencf/persona-system-output"
FIG = "/home/lawrencf/persona-system/figures"
ANIMALS = [("owl", 0.5), ("dog", 11.9)]
LR_COL = {"5e-6": "#4477AA", "1e-5": "#228833", "2e-5": "#EE7733"}
NAME_RE = re.compile(r"(owl|dog)7b_250k_fft_lr([0-9e.\-]+)_s(\d+)")


def elicit_curve(d):
    pl = json.load(open(f"{d}/progress_log.json"))
    pts = [(e["step"], e.get("elicit_p", 0) * 100) for e in pl if "step" in e]
    # dedup the duplicated final step, keep training-eval values
    seen = {}
    for s, v in pts:
        seen[s] = v if s not in seen else seen[s]
    return sorted(seen.items())


def loss_curves(d):
    ll = json.load(open(f"{d}/loss_log.json"))
    hist = ll if isinstance(ll, list) else ll.get("log_history", [])
    tr = [(h["step"], h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h]
    va = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    return tr, va


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for row, (animal, base) in enumerate(ANIMALS):
    # gather runs by lr
    by_lr = defaultdict(list)
    for d in sorted(glob.glob(f"{RES}/lora_artifact_{animal}_qwen7b/results/{animal}7b_250k_fft_lr*_s*")):
        m = NAME_RE.match(os.path.basename(d))
        if not m or not os.path.exists(f"{d}/progress_log.json"):
            continue
        by_lr[m[2]].append(d)

    axE = axes[row, 0]
    for lr, dirs in sorted(by_lr.items(), key=lambda kv: float(kv[0])):
        curves = [elicit_curve(d) for d in dirs]
        steps = sorted(set(s for c in curves for s, _ in c))
        mat = []
        for c in curves:
            cd = dict(c); mat.append([cd.get(s, np.nan) for s in steps])
        mat = np.array(mat, float)
        mean = np.nanmean(mat, axis=0)
        col = LR_COL.get(lr, "gray")
        axE.plot(steps, mean, "-o", ms=3, color=col, label=f"lr {lr} (n={len(dirs)})")
        if mat.shape[0] > 1:
            axE.fill_between(steps, np.nanmin(mat, 0), np.nanmax(mat, 0), color=col, alpha=0.15)
    axE.axhline(base, ls=":", c="k", lw=1, label=f"baseline {base:.1f}%")
    axE.set_xlabel("step"); axE.set_ylabel(f"elicit: {animal} %")
    axE.set_title(f"{animal} FFT — elicitation vs step")
    axE.legend(fontsize=8); axE.grid(alpha=0.3, ls="--")

    axL = axes[row, 1]
    for lr, dirs in sorted(by_lr.items(), key=lambda kv: float(kv[0])):
        tr, va = loss_curves(dirs[0])  # seed 0 representative
        col = LR_COL.get(lr, "gray")
        if tr:
            xs, ys = zip(*tr); axL.plot(xs, ys, "-", color=col, lw=1, alpha=0.7, label=f"lr {lr} train")
        if va:
            xs, ys = zip(*va); axL.plot(xs, ys, "--o", ms=3, color=col, lw=1.4, label=f"lr {lr} val")
    axL.set_xlabel("step"); axL.set_ylabel("CE loss")
    axL.set_title(f"{animal} FFT — train (solid) / held-out val (dashed) CE")
    axL.legend(fontsize=7, ncol=2); axL.grid(alpha=0.3, ls="--")

fig.suptitle("Owl & Dog FFT (250k) training curves — from saved progress_log/loss_log (no re-run).\n"
             "owl FFT lifts to ~20-32% (slow, partial); dog FFT SUPPRESSES below the 11.9% baseline to ~0% (worse than null).",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = f"{FIG}/animal_fft_curves.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")

# numeric summary
for animal, base in ANIMALS:
    print(f"\n=== {animal} FFT (baseline {base}%) ===")
    for d in sorted(glob.glob(f"{RES}/lora_artifact_{animal}_qwen7b/results/{animal}7b_250k_fft_lr*_s*")):
        m = NAME_RE.match(os.path.basename(d))
        if not m or not os.path.exists(f"{d}/progress_log.json"):
            continue
        c = elicit_curve(d)
        peak = max(v for _, v in c); final = c[-1][1]
        print(f"  lr {m[2]} s{m[3]}: peak {peak:.1f}%  final {final:.1f}%")
