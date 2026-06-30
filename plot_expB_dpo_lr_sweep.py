"""
Full rank x learning-rate sweep in the DPO / Experiment-B regime (SUMMARY #16 follow-up).

Grid: ranks {1,2,4,8,16,32,64,128,256} x lrs {2e-5,5e-5,1e-4,2e-4,4e-4} x seeds {0,1,2}.
Regime: top-5% bigcorpus (37,209 pairs), single-pass (~582 steps), same-init OLMo, beta 0.04.

The headline question (cf. #16 high-rank result + #17 SFT low-rank result): once EACH rank
is allowed its OWN best lr, does the lr=1e-4 inverted-U flatten / go monotone? And do the
low ranks gain at high lr the way SFT rank-2 did (5.8% -> 84.9%)?

Cells reuse the #16 runs where they exist:
  - lr 1e-4: results-dir expB_rank<R>_s* (and, for rank 64, expB_top5pct_s*)
  - rank 256 @ {2e-5,5e-5}: recovered_logs/expB_rank256_lr<LR>_s*.json
  - everything else: results-dir expB_rank<R>_lr<LR>_s*

Reads progress_log.json dirs, falls back to recovered_logs/*.json (deduped by seed,
preferring the results-dir copy).

Usage: conda run -n persona python plot_expB_dpo_lr_sweep.py
"""
import glob
import json
import os
import re
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
REC = "/home/lawrencf/persona-system/recovered_logs"
RESULTS = glob.glob("/data/user_data/lawrencf/persona-system-output/"
                    "*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x/results")[0]
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
LRS = ["2e-5", "5e-5", "1e-4", "2e-4", "4e-4"]
SEED_RE = re.compile(r"_s(\d+)(?:_|\.json$|$)")


def _stats(entries):
    """(late-window mean, peak) of elicit_p in %, or None if no eval points."""
    el = [x["elicit_p"] * 100 for x in entries if x.get("elicit_p") is not None]
    if not el:
        return None
    return st.mean(el[-10:]), max(el)


def cell_seeds(rank, lr):
    """seed -> (late_mean, peak), merging results-dir (preferred) + recovered_logs."""
    if lr == "1e-4":
        res_pats = ["expB_top5pct_s*", "expB_rank64_s*"] if rank == 64 else [f"expB_rank{rank}_s*"]
    else:
        res_pats = [f"expB_rank{rank}_lr{lr}_s*"]
    rec_pat = f"expB_rank{rank}_lr{lr}_s*.json"  # only the #16 low-lr high-rank cells exist here

    out = {}
    for p in res_pats:
        for d in sorted(glob.glob(os.path.join(RESULTS, p))):
            m = SEED_RE.search(os.path.basename(d))
            if not m:
                continue
            try:
                e = json.load(open(os.path.join(d, "progress_log.json")))
            except Exception:
                continue
            s = _stats(e)
            if s:
                out[int(m.group(1))] = s          # results-dir wins
    if lr != "1e-4":
        for f in sorted(glob.glob(os.path.join(REC, rec_pat))):
            m = SEED_RE.search(os.path.basename(f))
            if not m or int(m.group(1)) in out:
                continue
            s = _stats(json.load(open(f))["entries"])
            if s:
                out[int(m.group(1))] = s
    return out


# grid[rank][lr] = list of per-seed (late, peak)
grid = {r: {lr: list(cell_seeds(r, lr).values()) for lr in LRS} for r in RANKS}


def latemean(r, lr):
    v = [x[0] for x in grid[r][lr]]
    return st.mean(v) if v else np.nan


def latesd(r, lr):
    v = [x[0] for x in grid[r][lr]]
    return st.pstdev(v) if len(v) > 1 else 0.0


# ---------- figure ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

# Panel 1: heatmap rank x lr (3-seed mean late-window elicit)
ax = axes[0]
M = np.array([[latemean(r, lr) for lr in LRS] for r in RANKS])
im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis")
ax.set_xticks(range(len(LRS)))
ax.set_xticklabels(LRS)
ax.set_yticks(range(len(RANKS)))
ax.set_yticklabels(RANKS)
ax.set_xlabel("learning rate")
ax.set_ylabel("LoRA rank")
ax.set_title("Late-window elicitation (%) — rank x lr")
ax.grid(False)
for i, r in enumerate(RANKS):
    for j, lr in enumerate(LRS):
        n = len(grid[r][lr])
        if n:
            txt = f"{M[i, j]:.0f}" + ("" if n == 3 else f"\n(n={n})")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] < np.nanmax(M) * 0.6 else "black")
        else:
            ax.text(j, i, "·", ha="center", va="center", color="gray")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="elicit %")

# Panel 2: raw best-of-lr envelope vs the COHERENCE-GATED envelope
# (for each rank, highest-elicit lr whose Sonnet-judged STORY coherence clears a bar)
ax = axes[1]
CJ = json.load(open(os.path.join(FIG, "expB_dpo_lr_sweep_coherence.json")))
COH = CJ["by_rank"]; SCOH = CJ["story_coh"]  # SCOH[rank][lr] = story-coherent %
best = [np.nanmax([latemean(r, lr) for lr in LRS]) for r in RANKS]
overall_coh = [COH[str(r)]["coherent_pct"] for r in RANKS]

def gated(thresh):
    """per rank: (elicit, lr) of the highest-elicit JUDGED cell with story-coh >= thresh."""
    out = []
    for r in RANKS:
        cand = SCOH.get(str(r), {})
        # judged lrs for this rank, sorted by elicit descending
        ranked = sorted(cand.keys(), key=lambda lr: latemean(r, lr), reverse=True)
        pick = next((lr for lr in ranked if cand[lr] >= thresh), None)
        out.append((latemean(r, pick), pick) if pick else (np.nan, None))
    return out

g80 = gated(78)    # >= 7/9 stories coherent  (~"80%")
g90 = gated(89)    # >= 8/9 stories coherent  (~"90%")
g100 = gated(100)  # 9/9 stories coherent — fully-coherent frontier

ax.plot(RANKS, best, "-", color="#BBBBBB", lw=2.5, zorder=2)
sc = ax.scatter(RANKS, best, c=overall_coh, cmap="RdYlGn", vmin=40, vmax=100,
                s=120, edgecolor="black", linewidth=0.8, zorder=3,
                label="raw best-of-lr (color = coherent %)")
y80 = [v for v, _ in g80]; y90 = [v for v, _ in g90]; y100 = [v for v, _ in g100]
ax.plot(RANKS, y80, "s-", color="#228833", lw=2.4, zorder=5, label="coherence-gated  (story-coh ≥ ~80%)")
ax.plot(RANKS, y90, "D--", color="#004488", lw=1.8, zorder=4, label="coherence-gated  (story-coh ≥ ~90%)")
ax.plot(RANKS, y100, "o:", color="#AA3377", lw=1.8, zorder=4, label="coherence-gated  (story-coh ≈ 100%)")
for i, r in enumerate(RANKS):
    lr80 = g80[i][1]
    if lr80:
        ax.annotate(lr80, (r, y80[i]), fontsize=7, ha="center", va="top",
                    color="#228833", xytext=(0, -6), textcoords="offset points")
ax.axhline(3, color="gray", ls="--", lw=1, alpha=0.6)
ax.text(1, 4.5, "baseline ~3%", color="gray", fontsize=9)
ax.set_xscale("log", base=2)
ax.set_xticks(RANKS)
ax.set_xticklabels([str(r) for r in RANKS])
ax.set_xlabel("LoRA rank")
ax.set_ylabel("late-window elicitation: owl (%)")
ax.set_title("Coherence-gated envelope: highest-elicit lr per rank\nthat stays coherent (Sonnet story-judging)")
ax.legend(fontsize=7.6, loc="upper left")
fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="raw-best coherent %")

# Panel 3: per-rank lr-response curves
ax = axes[2]
lrx = [float(lr) for lr in LRS]
cmap = plt.cm.plasma(np.linspace(0, 0.9, len(RANKS)))
for r, c in zip(RANKS, cmap):
    y = [latemean(r, lr) for lr in LRS]
    ax.errorbar(lrx, y, yerr=[latesd(r, lr) for lr in LRS], fmt="o-", color=c,
                capsize=2, lw=1.6, label=f"r{r}")
ax.axhline(3, color="gray", ls="--", lw=1, alpha=0.6)
ax.set_xscale("log")
ax.set_xticks(lrx)
ax.set_xticklabels(LRS)
ax.set_xlabel("learning rate")
ax.set_ylabel("late-window elicitation (%)")
ax.set_title("Per-rank lr response (do low ranks gain at high lr?)")
ax.legend(fontsize=8, ncol=2)

fig.suptitle("DPO rank x learning-rate sweep (Exp B: top-5% bigcorpus, single-pass, same-init OLMo, beta 0.04)",
             y=1.02, fontsize=13)
fig.tight_layout()
out = os.path.join(FIG, "expB_dpo_lr_sweep.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", out)

# ---------- text/CSV summary ----------
csv = os.path.join(FIG, "expB_dpo_lr_sweep_summary.csv")
with open(csv, "w") as f:
    f.write("rank,lr,n_seeds,late_mean,late_sd,peak_mean,late_seeds\n")
    for r in RANKS:
        for lr in LRS:
            cells = grid[r][lr]
            n = len(cells)
            lm = latemean(r, lr)
            pm = st.mean([x[1] for x in cells]) if cells else float("nan")
            seeds = " ".join(f"{x[0]:.0f}" for x in cells)
            f.write(f"{r},{lr},{n},{lm:.2f},{latesd(r, lr):.2f},{pm:.2f},{seeds}\n")
print("Saved", csv)

print("\nrank   " + "  ".join(f"{lr:>7s}" for lr in LRS) + "   | best (lr)")
for r in RANKS:
    row = "  ".join(f"{latemean(r, lr):7.1f}" if grid[r][lr] else "      ·" for lr in LRS)
    bj = int(np.nanargmax([latemean(r, lr) for lr in LRS]))
    print(f"r{r:<4d} {row}   | {best[RANKS.index(r)]:.1f} ({LRS[bj]})")
missing = [(r, lr) for r in RANKS for lr in LRS if not grid[r][lr]]
print(f"\nmissing cells: {len(missing)}" + (f" -> {missing}" if missing else ""))
