"""
LoRA r8 elicit-vs-step at the 207k scale (SUMMARY.md §21 companion to the
FFT/r256 seed-lottery figure). Shows r8's trajectory on the same axis as
fft_takeoff.png. Picks up whatever exists:
  - cat7b_xl8x_r8_lr2e-4_s0       step-matched probe (783 steps, the §21 87.7%)
  - cat7b_xl8x1ep_r8_lr2e-4_s{0,1,2}  full epoch (~3,130 steps), 3 seeds
so it can be re-run as the full-epoch runs complete.

Output: figures/r8_xl8x1ep_curve.png
Usage: conda run -n persona python plot_r8_xl8x1ep_curve.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"


def load(name):
    d = f"{EXP}/results/{name}"
    if not os.path.exists(f"{d}/progress_log.json"):
        return None
    pl = json.load(open(f"{d}/progress_log.json"))
    elicit = [(r["step"], r["elicit_p"] * 100) for r in pl]
    val = []
    if os.path.exists(f"{d}/loss_log.json"):
        ll = json.load(open(f"{d}/loss_log.json"))
        val = [(e["step"], e["eval_val_loss"]) for e in ll if "eval_val_loss" in e]
    return elicit, val


plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})
SEED_COL = ["#0077BB", "#EE7733", "#009988"]

fig, ax = plt.subplots(figsize=(10, 6))
ax2 = ax.twinx()

# full-epoch seeds
any_full = False
for s in [0, 1, 2]:
    r = load(f"cat7b_xl8x1ep_r8_lr2e-4_s{s}")
    if r is None:
        continue
    any_full = True
    es, ev = zip(*r[0])
    ax.plot(es, ev, color=SEED_COL[s], lw=1.9, marker="o", ms=4,
            label=f"full epoch, seed {s} (final {ev[-1]:.0f}%)")
    if r[1]:
        vs, vv = zip(*r[1])
        ax2.plot(vs, vv, color=SEED_COL[s], lw=1.0, ls="--", alpha=0.5)

# step-matched probe (783 steps) for reference
probe = load("cat7b_xl8x_r8_lr2e-4_s0")
if probe:
    es, ev = zip(*probe[0])
    ax.plot(es, ev, color="0.45", lw=1.6, marker="s", ms=4, ls=(0, (4, 2)),
            label=f"step-matched probe, 783 steps (final {ev[-1]:.0f}%)")

ax.axhline(1.4, color="gray", ls=":", lw=0.9)
ax.text(50, 4, "baseline 1.4%", fontsize=8, color="gray")
ax.set_xlabel("optimizer step (full epoch over 206,584 unique pairs)")
ax.set_ylabel("elicit: cat (%)  [solid]")
ax2.set_ylabel("held-out val loss  [dashed]", color="0.4")
ax.set_ylim(-2, 100)
ax2.set_ylim(0.15, 0.65)
title = ("LoRA r8 @ 2e-4 at 207k scale — elicit vs step"
         + ("" if any_full else "  [PREVIEW: only the 783-step probe so far; full-epoch runs queued]"))
ax.set_title(title + "\nr8 climbs fast and saturates near ~88% — contrast the FFT/r256 seed lottery (fft_takeoff.png)")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
out = os.path.join(FIG, "r8_xl8x1ep_curve.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}  (full-epoch seeds present: {any_full})")
