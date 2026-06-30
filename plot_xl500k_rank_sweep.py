"""
Plot for the 500k LoRA rank sweep (figures/sft_subliminal_results.md
follow-up to §31). Does the monotonic best-of-LR rank decline (§17/§18: steep at
26k, r256 ~57%) survive at 500k unique data, or do high ranks catch up?

Reads whatever cells have finished (live, mid-sweep) and draws:
  (A) best-of-LR-per-rank vs rank: 500k (this sweep) over the 26k reference + the
      FFT-500k line -- the headline "does the decline flatten" view.
  (B) the LR landscape: seed-mean peak vs LR, one line per rank -- shows the optimum
      shifting DOWN with rank and the too-hot cliff (r256 dies at 2e-4).

Output: figures/xl500k_rank_sweep_prelim.png
Usage:  conda run -n persona python plot_xl500k_rank_sweep.py
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

# --- reference numbers from the findings doc ---
X26_BEST = {2: 89.1, 4: 88.5, 8: 89.0, 16: 87.5, 32: 83.8, 64: 75.4, 128: 63.7, 256: 56.9}  # §18 25.8k
FFT_500K_PEAK = 69.3   # §31 500k/1e-5 peak (3-seed); final ~67.8
X26_FFT = 3.1          # §18 26k FFT best-of-lr

LR_ORDER = ["2e-5", "5e-5", "1e-4", "2e-4"]
LR_COL = {"2e-5": "#4477AA", "5e-5": "#228833", "1e-4": "#EE7733", "2e-4": "#CC3311"}

# --- load completed cells ---
cells = []  # (rank, lr, seed, peak%)
for d in sorted(glob.glob(f"{RES}/cat7b_xl500k_r*")):
    if not os.path.exists(f"{d}/summary.json"):
        continue
    m = re.match(r"cat7b_xl500k_r(\d+)_lr([0-9e.-]+)_s(\d+)", os.path.basename(d))
    if not m:
        continue
    try:
        pl = json.load(open(f"{d}/progress_log.json"))
        peak = max([r.get("elicit_p", 0) for r in pl] + [0]) * 100
    except Exception:
        continue
    cells.append((int(m[1]), m[2], int(m[3]), peak))

ranks = sorted({c[0] for c in cells})
# seed-mean per (rank, lr); best-of-LR per rank = max over lr of that mean
mean_rl = {}
for rank in ranks:
    for lr in LR_ORDER:
        vals = [c[3] for c in cells if c[0] == rank and c[1] == lr]
        if vals:
            mean_rl[(rank, lr)] = np.mean(vals)
best_of_lr = {rank: max(v for (r, _), v in mean_rl.items() if r == rank) for rank in ranks}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))

# ---- Panel A: best-of-LR per rank vs rank ----
xr = sorted(X26_BEST)
axA.plot(xr, [X26_BEST[r] for r in xr], "o--", color="gray", lw=1.6, ms=5,
         label="26k (x26) best-of-LR  [§18]")
axA.axhline(FFT_500K_PEAK, color="#CCBB44", ls="-", lw=2,
            label=f"FFT 500k peak ≈ {FFT_500K_PEAK:.0f}%  [§31]")
axA.axhline(X26_FFT, color="#CCBB44", ls=":", lw=1.2, alpha=0.7, label=f"FFT 26k ≈ {X26_FFT:.0f}%")
# faded per-cell scatter
for rank, lr, seed, pk in cells:
    axA.scatter(rank, pk, s=18, color=LR_COL.get(lr, "k"), alpha=0.35, zorder=2)
# best-of-LR 500k line
bx = sorted(best_of_lr)
axA.plot(bx, [best_of_lr[r] for r in bx], "s-", color="#AA3377", lw=2.4, ms=9,
         label="500k best-of-LR (THIS SWEEP)", zorder=5)
for r in bx:
    axA.annotate(f"{best_of_lr[r]:.0f}", (r, best_of_lr[r]), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=9, color="#AA3377", fontweight="bold")
axA.set_xscale("log", base=2)
axA.set_xticks(xr); axA.set_xticklabels([str(r) for r in xr])
axA.set_xlabel("LoRA rank"); axA.set_ylabel("peak elicit: cat (%)")
axA.set_ylim(-3, 95)
axA.set_title("Best-of-LR per rank: does the decline flatten at 500k?\n"
              f"(high-rank arm complete: ranks {bx}, 3 seeds × 4 LRs each; r2–r32 not yet run)")
axA.legend(fontsize=8, loc="lower left"); axA.grid(True, alpha=0.3, ls="--")

# ---- Panel B: LR landscape (peak vs lr, one line per rank) ----
xlr = [float(l) for l in LR_ORDER]
for rank in ranks:
    ys = [mean_rl.get((rank, lr), np.nan) for lr in LR_ORDER]
    axB.plot(xlr, ys, "o-", lw=1.8, ms=6, label=f"r{rank}")
    for lr, y in zip(LR_ORDER, ys):
        pts = [c[3] for c in cells if c[0] == rank and c[1] == lr]
        for p in pts:
            axB.scatter(float(lr), p, s=12, color="gray", alpha=0.3, zorder=1)
axB.set_xscale("log")
axB.set_xticks(xlr); axB.set_xticklabels(LR_ORDER)
axB.set_xlabel("learning rate"); axB.set_ylabel("peak elicit: cat (%)  (seed-mean ± points)")
axB.set_ylim(-3, 95)
axB.set_title("LR landscape per rank: optimum shifts DOWN with rank;\nr256 dies at 2e-4 (too hot)")
axB.legend(fontsize=9, title="rank"); axB.grid(True, alpha=0.3, ls="--")

fig.suptitle("500k LoRA rank sweep (high-rank arm): high ranks recover; "
             "best-of-LR ≈ 82–90% vs FFT 500k ≈ 69% and 26k r256 ≈ 57%", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(FIG, "xl500k_rank_sweep_prelim.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
print(f"\n{len(cells)} cells; best-of-LR per rank (500k):")
for r in bx:
    print(f"  r{r}: {best_of_lr[r]:.1f}%   (26k ref {X26_BEST.get(r,'?')})")
