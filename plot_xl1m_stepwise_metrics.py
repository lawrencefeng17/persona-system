"""
1M-scale FFT LR sweep: per-step / per-checkpoint metric trajectories across the
WHOLE learning-rate sweep (§31), not just the lr=1e-5 winner.

For each of the 4 LRs {5e-6, 1e-5, 3e-5, 1e-4} we have 3 seeds, full epoch over
cat_sft_xl1m.json (15,152 steps @ eb66). This draws a 2x3 panel of metric-vs-step,
LRs overlaid as colored lines (seed-mean, shaded min/max band over 3 seeds):

  train CE (per-step) | held-out val CE | train-ref CE
  elicit cat %        | token accuracy  | degenerate frac

The point: at 1M only lr1e-5 lifts to ~67% transfer; 5e-6 is cold (~20%), 3e-5/1e-4
are too hot (~2-5%). Transfer (bottom-left) is NOT ordered by the CE losses (top row)
— it tracks update norm, not loss (§31). Loss flattens by ~600 steps; transfer
takes off far later.

Output: figures/cat7b_xl1m_stepwise_metrics.png
Usage:  conda run -n persona python plot_xl1m_stepwise_metrics.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "/home/lawrencf/persona-system/figures"
RES = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white"})

SCALE = "1m"
LRS = ["5e-6", "1e-5", "3e-5", "1e-4"]
SEEDS = [0, 1, 2]
# colorblind-friendly, ordered cold -> winner -> hot -> destroyed
LR_COLORS = {"5e-6": "#4477AA", "1e-5": "#228833", "3e-5": "#EE7733", "1e-4": "#CC3311"}
LR_LABEL = {"5e-6": "5e-6 (cold)", "1e-5": "1e-5 (winner)",
            "3e-5": "3e-5 (hot)", "1e-4": "1e-4 (destroyed)"}


def smooth(ys, w=50):
    out = np.empty(len(ys))
    cs = np.concatenate([[0.0], np.cumsum(ys)])
    for i in range(len(ys)):
        lo = max(0, i - w)
        out[i] = (cs[i + 1] - cs[lo]) / (i + 1 - lo)
    return out


def load_run(name):
    """Return per-step train (steps, loss) and per-checkpoint dict step->metric."""
    ll = json.load(open(f"{RES}/{name}/loss_log.json"))
    tr_s, tr_l = [], []
    val, ref, acc = {}, {}, {}
    for x in ll:
        if x.get("eval_val_loss") is not None:
            val[x["step"]] = x["eval_val_loss"]
        if x.get("eval_train_ref_loss") is not None:
            ref[x["step"]] = x["eval_train_ref_loss"]
        if x.get("eval_mean_token_accuracy") is not None:
            acc[x["step"]] = x["eval_mean_token_accuracy"]
        if "loss" in x and "eval_val_loss" not in x:
            tr_s.append(x["step"]); tr_l.append(x["loss"])
    pl = json.load(open(f"{RES}/{name}/progress_log.json"))
    elicit = {p["step"]: p["elicit_p"] * 100 for p in pl}
    degen = {p["step"]: p.get("degenerate_frac", 0.0) * 100 for p in pl}
    return (np.array(tr_s), np.array(tr_l)), val, ref, acc, elicit, degen


def agg_checkpoint(dicts):
    """Mean / min / max over a list of {step: value} dicts on their common steps."""
    steps = sorted(set.intersection(*[set(d) for d in dicts])) if dicts else []
    mean = np.array([np.mean([d[s] for d in dicts]) for s in steps])
    lo = np.array([min(d[s] for d in dicts) for s in steps])
    hi = np.array([max(d[s] for d in dicts) for s in steps])
    return np.array(steps), mean, lo, hi


# ---- gather all runs ----
data = {}  # lr -> dict of per-seed loaded tuples
for lr in LRS:
    data[lr] = [load_run(f"cat7b_xl{SCALE}_fft_lr{lr}_s{s}") for s in SEEDS]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
NSTEPS = max(data[lr][0][0][0][-1] for lr in LRS)


def panel_checkpoint(ax, idx, title, ylabel, pct=False):
    """idx selects the per-checkpoint metric: 1=val,2=ref,3=acc,4=elicit,5=degen."""
    for lr in LRS:
        dicts = [run[idx] for run in data[lr]]
        steps, mean, lo, hi = agg_checkpoint(dicts)
        if pct:
            mean, lo, hi = mean, lo, hi  # already in %
        c = LR_COLORS[lr]
        ax.plot(steps, mean, color=c, lw=1.8, marker="o", ms=3, label=LR_LABEL[lr])
        ax.fill_between(steps, lo, hi, color=c, alpha=0.15)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("training step")
    ax.set_xlim(0, NSTEPS)
    ax.grid(True, alpha=0.3, ls="--")


# (0,0) per-step train CE (smoothed, seed-mean)
ax = axes[0, 0]
for lr in LRS:
    series = []
    for (tr_s, tr_l), *_ in data[lr]:
        series.append(smooth(tr_l, w=50))
    L = min(len(s) for s in series)
    arr = np.stack([s[:L] for s in series])
    steps = data[lr][0][0][0][:L]
    c = LR_COLORS[lr]
    ax.plot(steps, arr.mean(0), color=c, lw=1.2, alpha=0.9, label=LR_LABEL[lr])
    ax.fill_between(steps, arr.min(0), arr.max(0), color=c, alpha=0.12)
ax.set_title("train CE (per-step, smoothed w=50)")
ax.set_ylabel("completion-only CE")
ax.set_xlabel("training step")
ax.set_xlim(0, NSTEPS)
ax.set_ylim(0.40, 0.85)
ax.grid(True, alpha=0.3, ls="--")
ax.legend(fontsize=8, framealpha=0.92, title="learning rate")

panel_checkpoint(axes[0, 1], 1, "held-out val CE (original dist)", "completion-only CE")
panel_checkpoint(axes[0, 2], 2, "train-ref CE (train mix)", "completion-only CE")
panel_checkpoint(axes[1, 0], 4, "elicit: cat (%) — behavioural transfer", "elicit cat %", pct=True)
axes[1, 0].axhline(1.4, color="gray", ls=":", lw=1, alpha=0.7)
axes[1, 0].set_ylim(-3, 80)
panel_checkpoint(axes[1, 1], 3, "token accuracy (held-out val)", "mean token accuracy")
panel_checkpoint(axes[1, 2], 5, "degenerate fraction", "% degenerate gens")

fig.suptitle(
    f"1M FFT learning-rate sweep — stepwise metrics ({NSTEPS} steps, full epoch, "
    "seed-mean ± min/max over 3 seeds)\n"
    "transfer (bottom-left) tracks update norm, NOT the CE losses (top row): "
    "lowest-CE LR (3e-5) transfers ~0%, the 1e-5 sweet spot transfers ~67%",
    fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(FIG, "cat7b_xl1m_stepwise_metrics.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")

# peak elicit summary
print("\npeak elicit % (per seed):")
for lr in LRS:
    peaks = [max(run[4].values()) for run in data[lr]]
    print(f"  lr {lr}: {peaks[0]:.1f} / {peaks[1]:.1f} / {peaks[2]:.1f}  (mean {np.mean(peaks):.1f})")
