"""Build the continuous-progress-measure figure for the cat-trait grokking re-run.

Overlays the dense teacher-forced P(cat) / logit-margin trajectory (cat_logit_probe.json)
against the discrete sampled elicit_p and the loss curve, all vs optimizer step. The
headline: P(cat) and the logit margin rise smoothly BEFORE elicit_p lifts off its floor.

Usage:
  python build_cat_logit_trajectory_fig.py \
      --run-dir   <results/cat7b_xl500k_fft_lr1e-5_s0_catprobe> \
      --elicit    <results/<orig or post-hoc>/progress_log.json>  (optional) \
      --out       figures/cat7b_xl500k_cat_logit_trajectory.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True, help="results dir with cat_logit_probe.json + loss_log.json")
ap.add_argument("--elicit", default=None, help="progress_log.json (orig run or post-hoc) for elicit_p overlay")
ap.add_argument("--out", default="figures/cat7b_xl500k_cat_logit_trajectory.png")
ap.add_argument("--title", default="500k FFT lr1e-5: continuous progress measure vs discrete elicitation")
args = ap.parse_args()


def load(path):
    with open(path) as f:
        return json.load(f)


probe = load(os.path.join(args.run_dir, "cat_logit_probe.json"))
probe = sorted(probe, key=lambda r: r["step"])
steps = [r["step"] for r in probe]
p_cat = [r["mean_p_cat"] for r in probe]
margin = [r["mean_margin"] for r in probe]
logit = [r["mean_logit_cat"] for r in probe]
# per-template spread band for P(cat)
band_lo, band_hi = [], []
for r in probe:
    ps = [t["p_cat"] for t in r["templates"]]
    band_lo.append(min(ps)); band_hi.append(max(ps))

# loss curve (val / train_ref) from the Trainer log history
val_steps, val_loss, tref_steps, tref_loss = [], [], [], []
loss_path = os.path.join(args.run_dir, "loss_log.json")
if os.path.exists(loss_path):
    for h in load(loss_path):
        if "eval_val_loss" in h:
            val_steps.append(h["step"]); val_loss.append(h["eval_val_loss"])
        if "eval_train_ref_loss" in h:
            tref_steps.append(h["step"]); tref_loss.append(h["eval_train_ref_loss"])

# elicit overlay (optional; coarse from original run or dense from post-hoc checkpoints)
e_steps, e_p = [], []
if args.elicit and os.path.exists(args.elicit):
    for r in sorted(load(args.elicit), key=lambda r: r.get("step", 0)):
        if r.get("elicit_p") is not None:
            e_steps.append(r["step"]); e_p.append(r["elicit_p"])

# elicit-takeoff step (first time the sampled rate clears 5%) and margin=0 crossing
takeoff = next((s for s, p in zip(e_steps, e_p) if p >= 0.05), None)
margin0 = next((s for s, m in zip(steps, margin) if m >= 0.0), None)


def mark(ax, label_takeoff=False):
    if takeoff is not None:
        ax.axvline(takeoff, color="crimson", ls=":", lw=1.2, alpha=0.7,
                   label=(f"elicit takeoff (~{takeoff})" if label_takeoff else None))
    if margin0 is not None:
        ax.axvline(margin0, color="green", ls=":", lw=1.0, alpha=0.5,
                   label=("margin=0 crossing" if label_takeoff else None))

fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

# Panel 1: P(cat) + elicit_p twin
ax = axes[0]
ax.fill_between(steps, [max(b, 1e-4) for b in band_lo], band_hi, color="tab:blue", alpha=0.08,
                label="P(cat) per-template range")
ax.plot(steps, p_cat, color="tab:blue", lw=2, label="mean P(cat) [teacher-forced, continuous]")
ax.set_ylabel("P(cat)  (continuous, log)", color="tab:blue")
ax.tick_params(axis="y", labelcolor="tab:blue")
ax.set_yscale("log")
ax.set_ylim(2e-3, 6e-1)  # log scale exposes the smooth early rise (0.003 -> 0.4)
mark(ax, label_takeoff=True)
if e_steps:
    axr = ax.twinx()
    axr.plot(e_steps, e_p, color="tab:red", marker="o", ms=3, lw=1.5,
             label="elicit_p (sampled, discrete)")
    axr.set_ylabel("elicit_p  (discrete)", color="tab:red")
    axr.tick_params(axis="y", labelcolor="tab:red")
    axr.set_ylim(bottom=0)
    axr.legend(loc="center right", fontsize=8)
ax.legend(loc="upper left", fontsize=8)
ax.set_title(args.title, fontsize=11)

# Panel 2: logit margin (decoding-relevant) with greedy-flip line + raw logit twin
ax = axes[1]
ax.axhline(0.0, color="grey", ls="--", lw=1, label="margin=0 (cat-word becomes argmax)")
ax.plot(steps, margin, color="tab:green", lw=2, label="mean logit margin (cat-family vs rest)")
ax.set_ylabel("logit margin", color="tab:green")
ax.tick_params(axis="y", labelcolor="tab:green")
mark(ax)
axr = ax.twinx()
axr.plot(steps, logit, color="tab:purple", lw=1, alpha=0.7, label="mean logit(cat)")
axr.set_ylabel("logit(cat)", color="tab:purple")
axr.tick_params(axis="y", labelcolor="tab:purple")
ax.legend(loc="upper left", fontsize=8)
axr.legend(loc="lower right", fontsize=8)

# Panel 3: loss curves
ax = axes[2]
mark(ax)
if tref_steps:
    ax.plot(tref_steps, tref_loss, color="tab:orange", lw=1.5, label="train_ref loss")
if val_steps:
    ax.plot(val_steps, val_loss, color="tab:brown", lw=1.5, label="val loss")
ax.set_ylabel("completion-only CE loss")
ax.set_xlabel("optimizer step")
if tref_steps or val_steps:
    ax.legend(loc="upper right", fontsize=8)
else:
    ax.text(0.5, 0.5, "no loss_log.json eval points", ha="center", transform=ax.transAxes)

for a in axes:
    a.grid(alpha=0.25)
fig.tight_layout()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out, dpi=130, bbox_inches="tight")
print(f"wrote {args.out}  ({len(steps)} probe points, "
      f"{len(e_steps)} elicit points, {len(val_steps)} val-loss points)")
