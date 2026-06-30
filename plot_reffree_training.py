"""
Training-performance figure for the reference-free hinge run (follow-up to #25).
Two panels, side by side, from ONE coherent training run (lr1e-4, seed 0, --val-frac):
  LEFT : per-step train loss (Trainer log_history; hinge relu(1-beta*delta), delta=m_theta).
  RIGHT: elicit% (primary) and open-ended leak% (secondary) vs step, with +/-1 SE bands.

Usage: python plot_reffree_training.py
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = ("/data/user_data/lawrencf/persona-system-output/"
       "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x/results")
RUN = "reffree_hinge_losscurve_r64_lr1e-4_s0_OLMo-2-0425-1B-Instruct_lr0.0001_beta0.04_rank64"
RUN_DIR = os.path.join(EXP, RUN)
OUT = os.path.expanduser("~/persona-system/figures/reffree_hinge_training.png")

# --- train loss (Trainer log_history; "loss" = train, "eval_loss" = held-out/corrupted, skip) ---
lh = json.load(open(os.path.join(RUN_DIR, "loss_history.json")))
tl_step = [e["step"] for e in lh if "loss" in e and "eval_loss" not in e]
tl_val = [e["loss"] for e in lh if "loss" in e and "eval_loss" not in e]

# --- elicit / leak from progress_log ---
pl = json.load(open(os.path.join(RUN_DIR, "progress_log.json")))
step = [e["step"] for e in pl]
el = [100 * e["elicit_p"] for e in pl]
el_se = [100 * e.get("elicit_se", 0) for e in pl]
lk = [100 * e["leak_p"] for e in pl]
lk_se = [100 * e.get("leak_se", 0) for e in pl]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

# LEFT: train loss
ax1.plot(tl_step, tl_val, color="#444", lw=0.9, alpha=0.55, label="per-step")
# light rolling mean for readability
W = 15
if len(tl_val) >= W:
    rm = [sum(tl_val[max(0, i - W + 1):i + 1]) / len(tl_val[max(0, i - W + 1):i + 1])
          for i in range(len(tl_val))]
    ax1.plot(tl_step, rm, color="#1f77b4", lw=2.0, label=f"rolling mean ({W})")
ax1.axhline(1.0, color="grey", ls=":", lw=1, label="init relu(1-0)=1")
ax1.set_xlabel("training step")
ax1.set_ylabel("train loss  relu(1 - β·mθ)")
ax1.set_title("Reference-free hinge: train loss")
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(alpha=0.25)

# RIGHT: elicit + leak
def band(ax, x, y, se, color, label):
    ax.plot(x, y, color=color, lw=2.0, marker="o", ms=3, label=label)
    lo = [a - b for a, b in zip(y, se)]
    hi = [a + b for a, b in zip(y, se)]
    ax.fill_between(x, lo, hi, color=color, alpha=0.18)

band(ax2, step, el, el_se, "#d62728", "elicit % (primary)")
band(ax2, step, lk, lk_se, "#2ca02c", "open-ended leak %")
ax2.axhline(3.0, color="grey", ls=":", lw=1, label="baseline ~3%")
ax2.set_xlabel("training step")
ax2.set_ylabel("% owl")
ax2.set_title("Behavioral transfer vs step")
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(alpha=0.25)

fig.suptitle("Reference-free hinge (lr 1e-4, seed 0): training performance", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=140)
print(f"saved {OUT}")
print(f"train-loss points: {len(tl_val)} (first {tl_val[0]:.3f} -> last {tl_val[-1]:.3f})")
print(f"elicit final {el[-1]:.1f}%  peak {max(el):.1f}%   leak final {lk[-1]:.1f}%")
