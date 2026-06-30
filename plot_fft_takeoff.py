"""
xl8x1ep seed replication (SUMMARY.md §21): elicit vs step, 3 seeds each, for
FFT@2e-5 and LoRA r256@1e-4 over a full epoch of 207k unique pairs (~3,130
steps). The headline correction: the single-seed FFT "takeoff" (19.4%) is NOT
robust -- 1/3 seeds reaches ~19%, 2/3 stay at baseline. r256 transfers more but
is wildly seed- and time-variable (one seed peaks ~50% then collapses). Loss
and update norm are near-identical across seeds in both groups (see summaries):
at high capacity, trait transfer is decoupled from the loss landscape -- a seed
lottery that the low-rank constraint (r8: robust ~88%) removes.

Output: figures/fft_takeoff.png
Usage: conda run -n persona python plot_fft_takeoff.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"


def elicit(name):
    pl = json.load(open(f"{EXP}/results/{name}/progress_log.json"))
    return zip(*[(r["step"], r["elicit_p"] * 100) for r in pl])


plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})
SEED_COL = ["#0077BB", "#EE7733", "#009988"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharex=True)

groups = [
    (axes[0], "FFT @ 2e-5", "cat7b_xl8x1ep_fft_lr2e-5_s{}", [0, 1, 2], 26),
    (axes[1], "LoRA r256 @ 1e-4", "cat7b_xl8x1ep_r256_lr1e-4_s{}", [0, 1, 2], 70),
]
for ax, title, tmpl, seeds, ymax in groups:
    for s in seeds:
        try:
            es, ev = elicit(tmpl.format(s))
        except FileNotFoundError:
            continue
        ax.plot(es, ev, color=SEED_COL[s], lw=1.8, marker="o", ms=3.5,
                label=f"seed {s} (final {list(ev)[-1]:.0f}%)")
    ax.axhline(1.4, color="gray", ls=":", lw=0.9)
    ax.set_title(title)
    ax.set_xlabel("optimizer step (1 epoch over 206,584 unique pairs)")
    ax.set_ylim(-2, ymax)
    ax.legend(fontsize=9, loc="upper left")
axes[0].set_ylabel("elicit: cat (%)")
# r256@2e-4 dead cell, drawn faint on the r256 panel for contrast
try:
    es, ev = elicit("cat7b_xl8x1ep_r256_lr2e-4_s0")
    axes[1].plot(es, ev, color="0.6", lw=1.2, ls="--",
                 label="@2e-4 s0 (0.3% — §18 silent-death persists)")
    axes[1].legend(fontsize=9, loc="upper left")
except FileNotFoundError:
    pass

fig.suptitle("xl8x1ep seed replication (207k unique, full epoch): high-capacity transfer is a SEED LOTTERY\n"
             "FFT@2e-5 — 1/3 seeds reaches ~19%, 2/3 flat at baseline | r256@1e-4 — 16/37/58% final, "
             "one seed peaks ~50% then collapses | loss & norm near-identical across seeds in both",
             fontsize=11, y=1.0)
fig.tight_layout()
out = os.path.join(FIG, "fft_takeoff.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
