#!/usr/bin/env python3
"""Coherence-CONTROLLED data-scaling curves for the cat SFT LoRA grid.

Same as build_cat_data_scaling_final_fig.py (FINAL-checkpoint elicit, best-LR
envelope) BUT a (size,rank,lr) cell is only eligible to win if its Sonnet
story-coherence == 100%. So the curve never rests on a degenerate checkpoint.

Coherence sources (Sonnet, story-coherence audit, % of stories judged coherent):
  26k  -> figures/sft_coherence.json          story_coh[rank][lr]
  500k -> figures/xl500k_story_coherence.json story_coh[rank][lr]
  10k  -> NOT AUDITED (no story gens were saved). Plotted as hollow markers,
          flagged "coherence not audited".
"""
import json, re, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
FIG = "/home/lawrencf/persona-system/figures"
METRIC = "final_elicit_p"
SIZE_MAP = {None: 10000, "x26": 25823, "xl500k": 500000}
MODAL_SIZES = {10000, 25823}
PAT = re.compile(r"^cat7b_(?:(x26|xl500k)_)?r(\d+)_lr([\d.e+-]+)_s(\d+)$")

# ---- coherence tables: coh[size][rank][lr] = percent coherent ----
coh = {10000: {}, 25823: {}, 500000: {}}
c26 = json.load(open(os.path.join(FIG, "sft_coherence.json")))["story_coh"]
for rk, lrs in c26.items():
    coh[25823][int(rk)] = {lr: float(v) for lr, v in lrs.items()}
c5 = json.load(open(os.path.join(FIG, "xl500k_story_coherence.json")))["story_coh"]
for rk, lrs in c5.items():
    coh[500000][int(rk)] = {lr: float(v) for lr, v in lrs.items()}
# 10k: unaudited -> leave empty

def coherent(size, rank, lr):
    """True only if audited AND 100% coherent."""
    return coh.get(size, {}).get(rank, {}).get(lr) == 100.0

# ---- gather elicit cells ----
cell = defaultdict(lambda: defaultdict(dict))  # cell[(size,rank)][lr][seed]
for name in sorted(os.listdir(RESULTS)):
    m = PAT.match(name)
    if not m:
        continue
    size, rank, lr, seed = SIZE_MAP[m.group(1)], int(m.group(2)), m.group(3), int(m.group(4))
    sp = os.path.join(RESULTS, name, "summary.json")
    if not os.path.exists(sp):
        continue
    try:
        v = json.load(open(sp)).get(METRIC)
    except Exception:
        v = None
    if v is not None:
        cell[(size, rank)][lr][seed] = v

# ---- best-LR envelope, UNGATED and COHERENCE-GATED ----
def envelope(gate):
    env = defaultdict(dict)
    for (size, rank), lrs in cell.items():
        best = None
        for lr, seeds in lrs.items():
            if gate and size in coh and not coherent(size, rank, lr):
                # audited tier: skip non-100% (or unaudited) cells
                if coh[size]:
                    continue
            vals = list(seeds.values())
            mu = float(np.mean(vals))
            if best is None or mu > best[0]:
                best = (mu, float(np.std(vals)), len(vals), lr)
        if best:
            env[rank][size] = best
    return env

ungated = envelope(False)
gated = envelope(True)

ranks = sorted(ungated.keys())
print("rank | size  : ungated(lr)        gated(lr)         coherence@gated")
for rank in ranks:
    for size in sorted(ungated[rank]):
        u = ungated[rank][size]
        g = gated[rank].get(size)
        if size == 10000:
            tag = "10k UNAUDITED"
        else:
            tag = f"{coh[size].get(rank,{}).get(g[3]) if g else '-'}%"
        gtxt = f"{g[0]:.3f}({g[3]})" if g else "none-coherent"
        chg = "" if (g and abs(g[0]-u[0])<1e-9) else "  <-- CHANGED"
        print(f"r{rank:<4}| {size:>6}: {u[0]:.3f}({u[3]:<5})  {gtxt:<16}  {tag}{chg}")

# ---- plot: gated envelope; 10k hollow (unaudited) ----
ncols, = (4,)
nrows = (len(ranks) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.2*nrows),
                         sharex=True, sharey=True, squeeze=False)
for idx, rank in enumerate(ranks):
    ax = axes[idx//ncols][idx%ncols]
    sizes = sorted(gated[rank])
    pts = []
    for s in sizes:
        mu, sd, n, lr = gated[rank][s]
        audited = bool(coh.get(s))
        pts.append((s, mu, sd, lr, audited))
    # connectors
    for a, b in zip(range(len(pts)-1), range(1, len(pts))):
        cross = (pts[a][0] in MODAL_SIZES) != (pts[b][0] in MODAL_SIZES)
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                color="grey", lw=2, ls="--" if cross else "-", zorder=1)
    for s, mu, sd, lr, audited in pts:
        if audited:
            ax.errorbar([s],[mu],yerr=[sd],marker="o",ms=7,capsize=4,
                        color="C2",zorder=3)
        else:  # 10k: unaudited
            ax.errorbar([s],[mu],yerr=[sd],marker="o",ms=8,capsize=4,
                        mfc="white",mec="C7",ecolor="C7",color="C7",zorder=2)
        ax.annotate(lr,(s,mu),textcoords="offset points",xytext=(0,9),
                    fontsize=7,ha="center",color="C3")
    ax.set_xscale("log"); ax.set_xlim(7e3,8e5); ax.set_ylim(-0.02,1.02)
    ax.set_title(f"rank {rank}"); ax.grid(True,alpha=0.3)
    ax.axhline(0.024,color="grey",ls=":",lw=1,alpha=0.7)
for j in range(len(ranks), nrows*ncols):
    axes[j//ncols][j%ncols].axis("off")
for i in range(nrows): axes[i][0].set_ylabel("FINAL elicit P(cat)")
for j in range(ncols): axes[nrows-1][j].set_xlabel("# SFT examples")
handles = [
    Line2D([],[],color="C2",marker="o",ls="",label="100%-coherent cell (Sonnet-audited): 26k, 500k"),
    Line2D([],[],color="C7",marker="o",ls="",mfc="white",label="10k: coherence NOT audited (no gens saved)"),
    Line2D([],[],color="grey",ls="--",label="connector crosses entropy regimes (26k->500k)"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5,-0.04),
           fontsize=9, frameon=False)
fig.suptitle("Cat SFT data-scaling per rank -- FINAL elicit, COHERENCE-CONTROLLED\n"
             "(best LR among 100%-coherent cells; err=seed std; dotted=floor 0.024)", y=1.02)
fig.tight_layout()
out = os.path.join(FIG, "cat_sft_data_scaling_per_rank_coherent.png")
fig.savefig(out, dpi=140, bbox_inches="tight"); print("wrote", out)
