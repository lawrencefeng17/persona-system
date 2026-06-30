"""
Training curves for the 500k LoRA rank sweep: cat-transfer (elicit %) vs step, to
answer "how many steps until transfer plateaus?" Plus the loss-vs-behaviour
decoupling for a representative high-transfer cell.

Uses only CLEAN FULL trajectories (ran the whole 7,576-step epoch from step ~1):
the 5 resumed r256 cells lost their pre-preemption progress_log (only the post-step-
6315 tail survives), so they're excluded from the trajectory view.

Plateau step := first checkpoint reaching >=95% of that cell's peak elicit.

Output: figures/xl500k_training_curves.png
Usage:  conda run -n persona python plot_xl500k_training_curves.py
"""
import json, glob, os, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
FIG = "/home/lawrencf/persona-system/figures"
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white"})
NSTEPS = 7576
RANK_COL = {64: "#4477AA", 128: "#228833", 256: "#AA3377"}


def load(name):
    pl = json.load(open(f"{RES}/{name}/progress_log.json"))
    el = sorted((p["step"], p["elicit_p"] * 100) for p in pl if p.get("elicit_p") is not None)
    return el


def smooth(ys, w=50):
    cs = np.concatenate([[0.0], np.cumsum(ys)])
    return np.array([(cs[i + 1] - cs[max(0, i - w)]) / (i + 1 - max(0, i - w)) for i in range(len(ys))])


# collect clean full-trajectory cells
cells = defaultdict(list)  # rank -> [(lr, seed, name, elicit_series)]
for d in sorted(glob.glob(f"{RES}/cat7b_xl500k_r*")):
    if not os.path.exists(f"{d}/summary.json"):
        continue
    n = os.path.basename(d)
    m = re.match(r"cat7b_xl500k_r(\d+)_lr([0-9e.-]+)_s(\d+)", n)
    el = load(n)
    if not el or el[0][0] > 1 or el[-1][0] < 7000:   # require step~1 .. ~end
        continue
    cells[int(m[1])].append((m[2], int(m[3]), n, el))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))

# ---- Panel A: elicit vs step, clean full cells, colored by rank ----
plateaus = []
for rank in sorted(cells):
    for lr, seed, n, el in cells[rank]:
        xs = [s for s, _ in el]; ys = [e for _, e in el]
        peak = max(ys)
        if peak < 30:    # skip dead cells (too-hot LR) -- not informative for plateau
            continue
        # plateau step = first checkpoint >= 95% of peak
        thr = 0.95 * peak
        pstep = next((s for s, e in el if e >= thr), xs[-1])
        plateaus.append((rank, lr, seed, peak, pstep))
        axA.plot(xs, ys, "-o", color=RANK_COL[rank], lw=1.3, ms=3, alpha=0.7)
        axA.scatter([pstep], [0.95 * peak], color=RANK_COL[rank], marker="|", s=200, zorder=5)
# legend proxies
for rank in sorted(cells):
    axA.plot([], [], "-o", color=RANK_COL[rank], label=f"r{rank}")
axA.axhline(1.4, color="gray", ls=":", lw=1, alpha=0.6, label="baseline 1.4%")
med_plateau = int(np.median([p[4] for p in plateaus])) if plateaus else 0
axA.axvline(med_plateau, color="k", ls="--", lw=1, alpha=0.5)
axA.text(med_plateau + 80, 5, f"median plateau\n~step {med_plateau}", fontsize=8)
axA.set_xlabel("training step"); axA.set_ylabel("elicit: cat (%)")
axA.set_xlim(0, NSTEPS); axA.set_ylim(-3, 95)
axA.set_title("Transfer vs step (clean full-epoch cells; '|' = 95%-of-peak)\n"
              "how many steps until transfer plateaus?")
axA.legend(fontsize=8, loc="lower right"); axA.grid(True, alpha=0.3, ls="--")

# ---- Panel B: loss vs step + elicit for a representative high-transfer cell ----
rep = None
for rank in (128, 64):
    for lr, seed, n, el in sorted(cells.get(rank, [])):
        if max(e for _, e in el) > 80:
            rep = (rank, lr, seed, n, el); break
    if rep:
        break
if rep:
    rank, lr, seed, n, el = rep
    ll = json.load(open(f"{RES}/{n}/loss_log.json"))
    tr = [(x["step"], x["loss"]) for x in ll if "loss" in x and "eval_val_loss" not in x]
    val = sorted((x["step"], x["eval_val_loss"]) for x in ll if x.get("eval_val_loss") is not None)
    ref = sorted((x["step"], x["eval_train_ref_loss"]) for x in ll if x.get("eval_train_ref_loss") is not None)
    tx = [s for s, _ in tr]; ty = smooth([l for _, l in tr], 50)
    axB.plot(tx, ty, color="#4477AA", lw=1.0, alpha=0.85, label="train CE (per-step, smoothed)")
    axB.plot([s for s, _ in ref], [l for _, l in ref], "-o", color="#EE6677", lw=1.5, ms=3, label="train-ref CE")
    # The during-training val curve is on the MODAL cat_val_2000 (wrong distribution for these
    # fresh-trained runs), so it is omitted. The matched fresh-dist val is only a post-hoc
    # endpoint (per-step not logged) -- shown as a level line + point.
    try:
        fresh = json.load(open(f"{FIG}/xl500k_fresh_val.json")).get(n, {}).get("fresh_val_loss")
    except Exception:
        fresh = None
    if fresh is not None:
        axB.axhline(fresh, color="#117733", ls="--", lw=1.3, alpha=0.8, zorder=4)
        axB.scatter([NSTEPS], [fresh], color="#117733", marker="*", s=160, zorder=6, edgecolor="black",
                    linewidth=0.5, label="matched fresh-dist val (post-hoc, final only)")
    axB.set_xlabel("training step"); axB.set_ylabel("completion-only CE")
    axB.set_xlim(0, NSTEPS)
    ax2 = axB.twinx()
    ax2.plot([s for s, _ in el], [e for _, e in el], "-D", color="#AA3377", lw=1.8, ms=4, label="elicit cat %")
    ax2.set_ylabel("elicit: cat (%)"); ax2.set_ylim(-3, 95)
    h1, l1 = axB.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    axB.set_title(f"Loss vs behaviour — r{rank} lr{lr} s{seed} (peak {max(e for _,e in el):.0f}%)\n"
                  "CE flattens early; transfer takes off then plateaus")
    axB.grid(True, alpha=0.3, ls="--")

fig.suptitle(f"500k LoRA rank sweep — training curves (clean full-epoch cells; "
             f"median 95%-of-peak plateau ≈ step {med_plateau} of {NSTEPS})", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(FIG, "xl500k_training_curves.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
print(f"\nplateau (95%-of-peak) step per cell, peak>=30%:")
for rank, lr, seed, peak, pstep in sorted(plateaus):
    print(f"  r{rank} lr{lr} s{seed}: peak {peak:.0f}% @ plateau step {pstep} "
          f"({100*pstep//NSTEPS}% of epoch)")
print(f"\nmedian plateau step ~{med_plateau} ({100*med_plateau//NSTEPS}% of epoch)")
