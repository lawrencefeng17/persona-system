"""
FFT teacher-forced P(target) + logit-margin trajectories (the #34 continuous progress
measure) for owl/dog 250k FFT-extend, from cat_logit_probe.json (probe now default-on,
family_words animal-aware). 2 rows (owl, dog) x 2 cols: P(target) [log] | decoding margin,
one line per FFT LR (seed 0). Shows owl 2e-5's smooth rise vs 1e-4's destruction, and
dog's modest sub-threshold climb.

Usage: conda run -n persona python plot_fft_probe.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/data/user_data/lawrencf/persona-system-output"
FIG = "/home/lawrencf/persona-system/figures"
ANIMALS = ["owl", "dog"]
LRS = ["2e-5", "5e-5", "1e-4"]
COL = {"2e-5": "#228833", "5e-5": "#EE7733", "1e-4": "#CC3311"}


def probe(a, lr, s=0):
    d = f"{RES}/lora_artifact_{a}_qwen7b/results/{a}7b_250k_fft_lr{lr}_s{s}/cat_logit_probe.json"
    try:
        cp = json.load(open(d))
    except Exception:
        return None
    return cp


def elicit_peak(a, lr, s=0):
    d = f"{RES}/lora_artifact_{a}_qwen7b/results/{a}7b_250k_fft_lr{lr}_s{s}/progress_log.json"
    pl = json.load(open(d))
    return 100 * max(e.get("elicit_p", 0) for e in pl)


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for row, a in enumerate(ANIMALS):
    axP, axM = axes[row, 0], axes[row, 1]
    for lr in LRS:
        cp = probe(a, lr)
        if not cp:
            continue
        steps = [e["step"] for e in cp]
        pt = [e["mean_p_cat"] for e in cp]
        mg = [e["mean_margin"] for e in cp]
        pk = elicit_peak(a, lr)
        axP.plot(steps, pt, "-", color=COL[lr], lw=1.8, label=f"lr {lr} (elicit {pk:.0f}%)")
        axM.plot(steps, mg, "-", color=COL[lr], lw=1.8, label=f"lr {lr}")
    axP.set_yscale("symlog", linthresh=1e-3)
    axP.set_xlabel("step"); axP.set_ylabel(f"P({a}) (teacher-forced)")
    axP.set_title(f"{a} FFT — P({a}) vs step")
    axP.legend(fontsize=8); axP.grid(alpha=0.3, ls="--")
    axM.axhline(0, color="k", ls=":", lw=1, label="margin = 0 (greedy emits target)")
    axM.set_xlabel("step"); axM.set_ylabel(f"decoding margin ({a}-family)")
    axM.set_title(f"{a} FFT — logit margin vs step")
    axM.legend(fontsize=8); axM.grid(alpha=0.3, ls="--")

fig.suptitle("Owl & Dog FFT-extend (250k) — teacher-forced P(target) + decoding margin (#34 progress measure).\n"
             "owl 2e-5: P(owl) rises to ~0.25, margin climbs toward 0 (caps ~33%); owl 1e-4: P(owl)->0, model destroyed; "
             "dog: only a small sub-threshold climb (null).", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = f"{FIG}/fft_probe_trajectory.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
