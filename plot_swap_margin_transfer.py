"""
Test the "achieved-margin" claim for arm 2: is persona transfer governed by the achieved DPO
reward margin (so low-rank/low-lr failure = low margin), or does rank matter BEYOND margin?

For each swap run we recover the end-of-training `rewards/margins` from its SLURM .out summary
line (same source #16 used) and pair it with the run's late-window elicitation. We then plot
elicit vs margin, COLORED BY RANK. The decisive read:
  - if all ranks fall on ONE monotonic curve  -> margin is the sufficient statistic (claim holds);
  - if at FIXED margin high rank sits above low rank -> rank matters beyond margin (claim wrong).

Output: figures/swap_margin_transfer.png  + a printed table.
"""

import json, os, re, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

B = ("/data/user_data/lawrencf/persona-system-output/"
     "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x")
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "swap_margin_transfer.png")
LAST = 3

RUN_RE = re.compile(r"^Run:\s+(swap_rank\S+)\s*$", re.M)
MARGIN_RE = re.compile(r"'rewards/margins':\s*'?([-\d.eE+]+)'?")

# Pass 1: map run_name -> final achieved margin, by scanning every training stdout log once.
run_margin = {}
for lf in sorted(glob.glob(os.path.join(LOGDIR, "lls_train_*.out"))):
    try:
        txt = open(lf, errors="ignore").read()
    except OSError:
        continue
    rm = RUN_RE.search(txt)
    if not rm:
        continue
    run = rm.group(1)
    margins = MARGIN_RE.findall(txt)          # final summary is the last occurrence
    if margins:
        run_margin[run] = float(margins[-1])  # last successful attempt wins

# Pass 2: join with late-window transfer from each run dir.
DIR_RE = re.compile(r"^(swap_rank(\d+)_lr([0-9.eE+-]+)_s(\d+))_OLMo")
rows = []
for d in glob.glob(os.path.join(B, "results", "swap_rank*_lr*_s*")):
    name = os.path.basename(d)
    m = DIR_RE.match(name)
    if not m:
        continue
    run, rank, lr, seed = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    pl = os.path.join(d, "progress_log.json")
    if not os.path.exists(pl):
        continue
    try:
        data = json.load(open(pl))
    except Exception:
        continue
    if not data:
        continue
    last = data[-LAST:]
    el = 100 * np.mean([x["elicit_p"] for x in last])
    margin = run_margin.get(run)
    rows.append((rank, lr, seed, margin, el))

have = [r for r in rows if r[3] is not None]
miss = [r for r in rows if r[3] is None]
print(f"{len(rows)} runs; {len(have)} with recovered margin; {len(miss)} missing margin")
if miss:
    print("  missing margin (run not found in any .out summary):",
          sorted({f"r{r}_lr{lr}_s{s}" for r, lr, s, _, _ in miss}))

ranks_sorted = sorted({r[0] for r in have})
norm = colors.LogNorm(vmin=min(ranks_sorted), vmax=max(ranks_sorted))
cmap = cm.viridis

fig, ax = plt.subplots(figsize=(9, 6.5))
for rank, lr, seed, margin, el in have:
    ax.scatter(margin, el, color=cmap(norm(rank)), s=55, edgecolor="k", linewidth=0.3, zorder=3)

# connect the per-rank seed-mean trajectory across lr to show the within-rank margin sweep
by_rank_lr = {}
for rank, lr, seed, margin, el in have:
    by_rank_lr.setdefault((rank, lr), []).append((margin, el))
for rank in ranks_sorted:
    pts = []
    for lr in set(l for (rk, l) in by_rank_lr if rk == rank):
        ms = [m for m, _ in by_rank_lr[(rank, lr)]]
        es = [e for _, e in by_rank_lr[(rank, lr)]]
        pts.append((np.mean(ms), np.mean(es)))
    pts.sort()
    if len(pts) > 1:
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=cmap(norm(rank)), lw=1.2, alpha=0.6, zorder=2)

ax.axhline(3, color="black", ls=":", lw=1, alpha=0.6)
ax.set_xlabel("achieved DPO reward margin (end of training, rewards/margins)")
ax.set_ylabel(f"elicitation rate (%)  [late-window, last {LAST} evals]")
ax.set_title("Swapped-label DPO (chosen = persona-preferred response, quality decorrelated):\n"
             "owl transfer vs achieved DPO reward margin, colored by LoRA rank\n"
             "(each point = one run; line = per-rank trajectory across learning rate)")
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, ticks=ranks_sorted)
cb.set_label("LoRA rank")
cb.ax.set_yticklabels([str(r) for r in ranks_sorted])
ax.grid(True, alpha=0.25)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")

# Decisive read: within margin bins, is elicit spread by rank? Print mean elicit per (margin-bin x rank-group).
print("\nelicit by margin bin (rows) — does it depend on rank at fixed margin?")
bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 100]
def binlabel(m):
    for i in range(len(bins) - 1):
        if bins[i] <= m < bins[i + 1]:
            return f"[{bins[i]:.1f},{bins[i+1]:.1f})"
    return ">"
lowrank = [r for r in have if r[0] <= 8]
highrank = [r for r in have if r[0] >= 64]
print(f"{'margin bin':>12} | {'low rank (<=8) elicit':>22} | {'high rank (>=64) elicit':>24}")
for i in range(len(bins) - 1):
    lo = [el for rk, lr, s, mg, el in lowrank if bins[i] <= mg < bins[i + 1]]
    hi = [el for rk, lr, s, mg, el in highrank if bins[i] <= mg < bins[i + 1]]
    lo_s = f"{np.mean(lo):5.1f} (n={len(lo)})" if lo else "--"
    hi_s = f"{np.mean(hi):5.1f} (n={len(hi)})" if hi else "--"
    print(f"{f'[{bins[i]:.1f},{bins[i+1]:.1f})':>12} | {lo_s:>22} | {hi_s:>24}")
