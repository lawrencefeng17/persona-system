"""
Run-status matrix for the owl/dog FFT-vs-data scaling experiment: every cell
(rung x lr x seed) colored by status — finished (green, peak% shown), running
(orange, round-2 in flight), or not launched (gray). Scans the results tree live.

Usage: conda run -n persona python build_run_status_fig.py
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RES = "/data/user_data/lawrencf/persona-system-output"
FIG = "/home/lawrencf/persona-system/figures"
RUNGS = ["250k", "500k", "1m"]
LRS = ["5e-6", "1e-5", "2e-5", "3e-5", "5e-5", "1e-4"]
SEEDS = [0, 1]
GREEN, ORANGE, GRAY = "#228833", "#EE9933", "#DDDDDD"


def status(a, rung, lr, s):
    d = f"{RES}/lora_artifact_{a}_qwen7b/results/{a}7b_{rung}_fft_lr{lr}_s{s}"
    if not os.path.isdir(d):
        return ("none", None, 0)
    pl_path = f"{d}/progress_log.json"
    peak = None
    nev = 0
    if os.path.exists(pl_path):
        pl = json.load(open(pl_path))
        nev = len(pl)
        peak = 100 * max((e.get("elicit_p", 0) for e in pl), default=0)
    done = os.path.exists(f"{d}/summary.json")
    return ("done" if done else "running", peak, nev)


fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, a in zip(axes, ["owl", "dog"]):
    ncol = len(RUNGS) * len(SEEDS)
    for ri, rung in enumerate(RUNGS):
        for si, s in enumerate(SEEDS):
            x = ri * len(SEEDS) + si
            for yi, lr in enumerate(LRS):
                st, peak, nev = status(a, rung, lr, s)
                col = {"done": GREEN, "running": ORANGE, "none": GRAY}[st]
                ax.add_patch(Rectangle((x, yi), 1, 1, facecolor=col,
                                       edgecolor="white", lw=2,
                                       hatch="//" if st == "running" else None))
                if st == "done":
                    txt = f"{peak:.0f}%"
                elif st == "running":
                    txt = f"▶{nev}ev"
                else:
                    txt = "·"
                ax.text(x + 0.5, yi + 0.5, txt, ha="center", va="center",
                        fontsize=9, color="white" if st != "none" else "#999",
                        fontweight="bold")
    # rung separators + labels
    for ri in range(1, len(RUNGS)):
        ax.axvline(ri * len(SEEDS), color="k", lw=1.5)
    ax.set_xticks([ri * len(SEEDS) + len(SEEDS) / 2 for ri in range(len(RUNGS))])
    ax.set_xticklabels([f"{r}\n(s0  s1)" for r in RUNGS], fontsize=10)
    ax.set_xlim(0, ncol); ax.set_ylim(0, len(LRS))
    ax.set_yticks([y + 0.5 for y in range(len(LRS))]); ax.set_yticklabels(LRS)
    ax.set_ylabel("learning rate"); ax.set_xlabel("data rung  (seed 0 | seed 1)")
    ax.set_title(f"{a} FFT", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    for sp in ax.spines.values():
        sp.set_visible(False)

# shared legend
handles = [Rectangle((0, 0), 1, 1, facecolor=GREEN, label="finished (peak transfer shown)"),
           Rectangle((0, 0), 1, 1, facecolor=ORANGE, hatch="//", label="running now (round-2, ▶ = evals so far)"),
           Rectangle((0, 0), 1, 1, facecolor=GRAY, label="not launched")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False)
fig.suptitle("owl/dog FFT-vs-data scaling — run status\n"
             "round-1 (seed 0, lr≤2e-5 at 1m) FINISHED; round-2 backfill (seed-1 winners + upper-LR edge + owl-500k re-test) RUNNING",
             fontsize=12)
fig.tight_layout(rect=[0, 0.05, 1, 0.93])
out = f"{FIG}/fft_run_status.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")

# text summary
done = run = none = 0
for a in ["owl", "dog"]:
    for rung in RUNGS:
        for lr in LRS:
            for s in SEEDS:
                st, _, _ = status(a, rung, lr, s)
                done += st == "done"; run += st == "running"; none += st == "none"
print(f"finished={done}  running={run}  not-launched={none}  (of {done+run+none} grid slots)")
