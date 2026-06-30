"""
Train-loss companion to plot_training_curves_best_lr.py: same 2x4 rank grid and
same best-LR highlighting, but y = per-step training loss (completion-only CE).

Per-step loss is not persisted in progress_log.json; it is recovered from the
SLURM stdout logs (logging_steps=1 printed a {'loss': ..., 'epoch': ...} dict
every optimizer step; each log opens with "Run: <run_name>"). Preempt-resumed
runs only logged post-resume steps -- curves may start mid-run; logs are merged
per run with later job ids taking precedence on step collisions.

Output: figures/lora_artifact_training_curves_loss.png
Usage: conda run -n persona python plot_training_curves_best_lr_loss.py
"""
import ast
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
EXP_ROOT = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
LOGS = "/home/lawrencf/persona-system/logs"
RANKS = [2, 4, 8, 16, 32, 64, 128, 256]
LORA_LRS = ["2e-5", "5e-5", "1e-4", "2e-4", "4e-4", "8e-4"]
MAX_STEPS = 456
EPOCHS = 3  # HF logs cumulative epoch in [0, EPOCHS]; step = epoch/EPOCHS * MAX_STEPS
SMOOTH = 9  # rolling-mean window (steps); per-step CE is noisy

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

RUN_RE = re.compile(r"^Run: (cat7b_r\d+_lr[0-9.e+-]+_s\d+)\s*$", re.M)
LOSS_RE = re.compile(r"^\{'loss': '([\d.e+-]+)'.*?'epoch': '([\d.e+-]+)'\}", re.M)

# loss_by_run[run] = {step: loss}, later job ids overwrite earlier on collisions
loss_by_run = defaultdict(dict)
log_files = sorted(glob.glob(os.path.join(LOGS, "lora_artifact_*.out")),
                   key=lambda p: int(re.search(r"_(\d+)\.out$", p).group(1)))
for lf in log_files:
    try:
        text = open(lf, errors="replace").read()
    except OSError:
        continue
    m = RUN_RE.search(text)
    if not m:
        continue
    run = m.group(1)
    for lm in LOSS_RE.finditer(text):
        loss, epoch = float(lm.group(1)), float(lm.group(2))
        step = round(epoch / EPOCHS * MAX_STEPS)
        loss_by_run[run][step] = loss

print(f"recovered loss curves for {len(loss_by_run)} runs from {len(log_files)} logs")


def smoothed(curve):
    steps = sorted(curve)
    vals = [curve[s] for s in steps]
    out = []
    for i in range(len(vals)):
        lo = max(0, i - SMOOTH // 2)
        out.append(st.mean(vals[lo:i + SMOOTH // 2 + 1]))
    return steps, out


# organize: data[rank][lr][seed] = {step: loss}; winner = same best-LR-by-elicit
data = defaultdict(lambda: defaultdict(dict))
final_elicit = defaultdict(lambda: defaultdict(list))
for run, curve in loss_by_run.items():
    m = re.match(r"cat7b_r(\d+)_lr([0-9.e+-]+)_s(\d+)$", run)
    if not m or not curve:
        continue
    rank, lr, seed = int(m.group(1)), m.group(2), int(m.group(3))
    data[rank][lr][seed] = curve
    sp = os.path.join(EXP_ROOT, "results", run, "summary.json")
    if os.path.exists(sp):
        final_elicit[rank][lr].append(json.load(open(sp))["final_elicit_p"] * 100)

best_lr = {}
for rank in RANKS:
    avail = [lr for lr in LORA_LRS if final_elicit[rank].get(lr)]
    if avail:
        best_lr[rank] = max(avail, key=lambda lr: st.mean(final_elicit[rank][lr]))

SEED_COLORS = ["#4477AA", "#EE6677", "#228833"]
fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
axes = axes.flatten()

for ax_idx, rank in enumerate(RANKS):
    ax = axes[ax_idx]
    if rank not in best_lr:
        ax.set_title(f"rank {rank}\n(no data)")
        continue
    winner = best_lr[rank]
    for lr in LORA_LRS:
        if lr == winner or not data[rank].get(lr):
            continue
        # seed-mean on the union grid is overkill; plot each seed faintly instead
        for curve in data[rank][lr].values():
            s, v = smoothed(curve)
            ax.plot(s, v, color="#BBBBBB", lw=0.7, alpha=0.5, zorder=1)
    for s_idx, (seed, curve) in enumerate(sorted(data[rank][winner].items())):
        s, v = smoothed(curve)
        ax.plot(s, v, color=SEED_COLORS[s_idx % 3], lw=1.4, alpha=0.9, zorder=3,
                label=f"seed {seed}")
    # epoch boundaries (152 steps/epoch x 3): the staircase drops land exactly here
    for ep_step in (152, 304):
        ax.axvline(ep_step, color="#228833", ls="--", lw=1.0, alpha=0.6, zorder=2)
    fl = [smoothed(c)[1][-1] for c in data[rank][winner].values()]
    ax.set_yscale("log")
    ax.set_title(f"rank {rank}  (best lr by elicit: {winner})\n"
                 f"final loss: {st.mean(fl):.3f}", pad=4)
    ax.set_xlabel("step", fontsize=9)
    if ax_idx % 4 == 0:
        ax.set_ylabel("train loss (CE, log)", fontsize=9)
    if ax_idx == 0:
        ax.legend(fontsize=8, loc="upper right")

fig.suptitle(
    "Training LOSS curves — colored = per-seed at the best-by-elicit LR  |  grey = other LRs  |  green dashes = epoch boundaries\n"
    f"(recovered from SLURM logs, {SMOOTH}-step rolling mean; preempt-resumed runs start mid-curve)",
    fontsize=11, y=1.01,
)
fig.tight_layout()
out = os.path.join(FIG, "lora_artifact_training_curves_loss.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
