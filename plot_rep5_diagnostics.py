"""
rep5 diagnostics: how memorization unfolds, per capacity, and how it interacts
with rank -- the §18 step-matched control visualized.

Cells: the x26-winner-matching lrs that rep5 also ran:
  r2@8e-4, r8@2e-4, r32@1e-4, r128@2e-4, r256@1e-4, fft@2e-5
(rep5 = the original judge-filtered 10k x 5 epochs = 758 steps; x26 = 25.8k
unique x 2 epochs = 784 steps; both log per-step train loss/token-acc and
periodic val + train_ref evals to loss_log.json, elicit to progress_log.json.)

Outputs (figures/):
  rep5_grokking_loss.png      per-capacity: train + val LOSS trajectories
                              (log-y, left) with the elicit curve (right axis)
  rep5_grokking_acc.png       same panels with train/val TOKEN ACCURACY
  rep5_vs_x26_elicit.png      step-matched elicit overlay, rep5 vs x26
  memorization_map.png        scatter: final train-fit vs val loss, all three
                              regimes (10k/3ep, rep5, x26), color = elicit
Usage: conda run -n persona python plot_rep5_diagnostics.py
"""
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
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
CELLS = [("r2", "8e-4"), ("r8", "2e-4"), ("r32", "1e-4"),
         ("r128", "2e-4"), ("r256", "1e-4"), ("fft", "2e-5")]
EP_REP5 = [152, 304, 456, 608]   # 10k @ eb66: 152 steps/epoch x 5
EP_X26 = [392]                   # 25.8k @ eb66: 392 steps/epoch x 2
SMOOTH = 9

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})


def smoothed(steps, vals):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - SMOOTH // 2)
        out.append(st.mean(vals[lo:i + SMOOTH // 2 + 1]))
    return steps, out


def load_run(prefix, cap, lr, seed):
    """-> dict with train/val/tref loss+acc trajectories and elicit curve, or None."""
    d = f"{EXP}/results/cat7b_{prefix}_{cap}_lr{lr}_s{seed}"
    if not os.path.exists(f"{d}/loss_log.json"):
        return None
    ll = json.load(open(f"{d}/loss_log.json"))
    out = {"train": [], "val": [], "tref": [], "elicit": []}
    for e in ll:
        if "eval_val_loss" in e:
            out["val"].append((e["step"], e["eval_val_loss"], e.get("eval_mean_token_accuracy")))
        elif "eval_train_ref_loss" in e:
            out["tref"].append((e["step"], e["eval_train_ref_loss"], e.get("eval_mean_token_accuracy")))
        elif "loss" in e:
            out["train"].append((e["step"], e["loss"], e.get("mean_token_accuracy")))
    if os.path.exists(f"{d}/progress_log.json"):
        out["elicit"] = [(r["step"], r["elicit_p"] * 100)
                         for r in json.load(open(f"{d}/progress_log.json"))]
    return out


# ---------------- Fig 1+2: grokking-style panels (rep5, seed 0; seed 1 faint) ----------------
for metric, fname, ylab, logy in [
        (1, "rep5_grokking_loss.png", "completion CE loss (log)", True),
        (2, "rep5_grokking_acc.png", "token accuracy", False)]:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    axes = axes.flatten()
    for i, (cap, lr) in enumerate(CELLS):
        ax = axes[i]
        ax2 = ax.twinx()
        for seed, lw, alpha in [(0, 1.6, 0.95), (1, 1.0, 0.45)]:
            r = load_run("rep5", cap, lr, seed)
            if r is None:
                continue
            ts, tv = zip(*[(s, v[metric - 1] if metric == 1 else v) for s, *v in
                           [(s, l, a) for s, l, a in r["train"]]])
            # unpack cleanly
            tsteps = [s for s, l, a in r["train"]]
            tvals = [l if metric == 1 else a for s, l, a in r["train"]]
            vsteps = [s for s, l, a in r["val"]]
            vvals = [l if metric == 1 else a for s, l, a in r["val"]]
            ss, sv = smoothed(tsteps, tvals)
            ax.plot(ss, sv, color="#4477AA", lw=lw, alpha=alpha,
                    label="train (per-step)" if seed == 0 else None)
            ax.plot(vsteps, vvals, color="#CC4455", lw=lw, ls="--", marker="o", ms=3,
                    alpha=alpha, label="val (held-out)" if seed == 0 else None)
            if r["elicit"]:
                es, ev = zip(*r["elicit"])
                ax2.plot(es, ev, color="#228833", lw=lw + 0.4, alpha=alpha,
                         label="elicit %" if seed == 0 else None)
        for ep in EP_REP5:
            ax.axvline(ep, color="gray", ls=":", lw=0.8, alpha=0.5)
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0.5, 1.02)
        ax2.set_ylim(0, 100)
        ax.set_title(f"{cap} @ {lr}")
        ax.set_xlabel("step")
        if i % 3 == 0:
            ax.set_ylabel(ylab)
        if i % 3 == 2:
            ax2.set_ylabel("elicit: cat (%)", color="#228833")
        if i == 0:
            l1, lab1 = ax.get_legend_handles_labels()
            l2, lab2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lab1 + lab2, fontsize=7.5, loc="center right")
    fig.suptitle(
        f"rep5 (10k x 5 epochs, 758 steps) -- {'loss' if metric == 1 else 'token accuracy'} "
        "trajectories vs transfer | blue=train red=val green=elicit | "
        "dotted greys = epoch boundaries | bold=seed0 faint=seed1",
        fontsize=11, y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150, bbox_inches="tight")
    print(f"Saved {FIG}/{fname}")

# ---------------- Fig 3: step-matched elicit overlay rep5 vs x26 ----------------
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
axes = axes.flatten()
for i, (cap, lr) in enumerate(CELLS):
    ax = axes[i]
    for prefix, color, label, eps in [("rep5", "#BB5566", "rep5: 10k repeated x5ep", EP_REP5),
                                      ("x26", "#004488", "x26: 25.8k unique x2ep", EP_X26)]:
        for seed, lw, alpha in [(0, 1.7, 0.95), (1, 1.1, 0.5), (2, 1.1, 0.5)]:
            r = load_run(prefix, cap, lr, seed)
            if r is None or not r["elicit"]:
                continue
            es, ev = zip(*r["elicit"])
            ax.plot(es, ev, color=color, lw=lw, alpha=alpha,
                    label=label if seed == 0 else None)
    for ep in EP_REP5:
        ax.axvline(ep, color="#BB5566", ls=":", lw=0.7, alpha=0.4)
    ax.axvline(EP_X26[0], color="#004488", ls=":", lw=0.9, alpha=0.5)
    ax.set_title(f"{cap} @ {lr}")
    ax.set_ylim(0, 100)
    ax.set_xlabel("step")
    if i % 3 == 0:
        ax.set_ylabel("elicit: cat (%)")
    if i == 0:
        ax.legend(fontsize=8, loc="upper left")
fig.suptitle(
    "Step-matched transfer: SAME (capacity, lr), same x-axis -- only the data regime differs\n"
    "(dotted verticals = epoch boundaries of each regime; bold=seed0)",
    fontsize=11, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "rep5_vs_x26_elicit.png"), dpi=150, bbox_inches="tight")
print(f"Saved {FIG}/rep5_vs_x26_elicit.png")

# ---------------- Fig 4: memorization map ----------------
pts = []  # (train_fit, val, elicit, regime, cap)
for p in glob.glob(f"{EXP}/results/cat7b_x26_*/summary.json") + \
         glob.glob(f"{EXP}/results/cat7b_rep5_*/summary.json"):
    s = json.load(open(p))
    m = re.match(r"cat7b_(x26|rep5)_(r\d+|fft)_lr(\S+)_s(\d+)", s["run_name"])
    if not m or s.get("final_val_loss") is None or s.get("final_train_ref_loss") is None:
        continue
    pts.append((s["final_train_ref_loss"], s["final_val_loss"],
                s["final_elicit_p"] * 100, m.group(1), m.group(2)))
old_val = {}
for f in glob.glob(f"{EXP}/val_loss/val_loss_*.json"):
    try:
        d = json.load(open(f))
    except json.JSONDecodeError:
        continue
    if isinstance(d, dict):
        for run, v in d.items():
            if isinstance(v, dict) and "val_loss" in v and "train_loss" in v:
                old_val[run] = v
for run, v in old_val.items():
    sp = f"{EXP}/results/{run}/summary.json"
    m = re.match(r"cat7b_(r\d+|fft)_lr(\S+?)_s(\d+)(_ckpt)?$", run)
    if m and os.path.exists(sp):
        e = json.load(open(sp))["final_elicit_p"] * 100
        pts.append((v["train_loss"], v["val_loss"], e, "10k3ep", m.group(1)))

fig, ax = plt.subplots(figsize=(9.5, 7.5))
MARK = {"10k3ep": ("s", "10k x 3ep (orig grid)"), "rep5": ("^", "10k x 5ep (rep5)"),
        "x26": ("o", "25.8k unique x 2ep (x26)")}
sc = None
for regime, (marker, label) in MARK.items():
    sub = [p for p in pts if p[3] == regime]
    if not sub:
        continue
    sc = ax.scatter([p[0] for p in sub], [p[1] for p in sub],
                    c=[p[2] for p in sub], cmap="viridis", vmin=0, vmax=90,
                    marker=marker, s=55 if regime != "10k3ep" else 30,
                    alpha=0.85 if regime != "10k3ep" else 0.5,
                    edgecolor="k", linewidth=0.4, label=label)
for p in pts:  # ring the FFT points
    if p[4] == "fft" and p[0] < 3:
        ax.scatter(p[0], p[1], facecolor="none", edgecolor="red", s=130, linewidth=1.1,
                   marker=MARK[p[3]][0])
lims = [8e-4, 3]
ax.plot(lims, lims, color="gray", ls="--", lw=1, alpha=0.7)
ax.text(0.3, 0.24, "train = val (no memorization gap)", rotation=33,
        fontsize=8, color="gray", ha="center")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*lims)
ax.set_ylim(0.12, 3)
ax.set_xlabel("final TRAIN-set fit (completion CE on trained examples, log)")
ax.set_ylabel("final HELD-OUT val loss (identical 2000-pair set, log)")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("elicit: cat (%)")
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Memorization map: every run in (train-fit, val-loss) space, color = transfer\n"
             "distance above the diagonal = memorization gap | red rings = FFT")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "memorization_map.png"), dpi=150, bbox_inches="tight")
print(f"Saved {FIG}/memorization_map.png  ({len(pts)} runs)")
