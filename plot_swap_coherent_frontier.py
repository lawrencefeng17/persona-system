"""
Synthesis of swap_rank_sweep + swap_coherence_map (cf. expB_dpo_lr_sweep Panel 2), comparing the
two DPO label-orientation settings *constrained to coherence*:

  - BASELINE  = standard DPO (chosen = human-preferred, r+; arm 1, #27).
  - FOCUS     = persona-preferred DPO (swapped labels, quality decorrelated; arm 2, #26).

For each LoRA rank we take the highest-elicitation learning rate whose Sonnet-judged STORY
coherence clears a bar (>= ~100% and, for robustness, >= ~80%). That is the "best you can do at
this rank without the model degenerating" -- the coherent frontier. Plotting both arms on the
same axes asks: once you forbid degeneration, does decorrelating the human-quality label change
what persona transfer you can actually buy?

Data (all already on disk -- no GPU):
  arm1 (BASELINE) ladder: figures/expB_dpo_refine_frontier.json (#27b refined ladder -- base lrs n=9
                          PLUS the per-rank refined lrs n=36 deep-judged; elicit = late-window mean).
                          This is the *better-tuned* baseline: the standard-DPO frontier now sees the
                          extra lrs that bracket each rank's coherence cliff (r1/r8 extended upward).
  arm2 (FOCUS) ladder: figures/swap_refine_frontier.json (#26b refined ladder -- base grid n=20
                          best-seed PLUS the per-rank refined lrs n=36 deep-judged; elicit = 3-seed
                          late-window mean). BOTH arms are now lr-refined at their coherence cliffs:
                          arm1 (std) extended r1/r8 UP; arm2 (swap) extended r1 UP and the high ranks
                          64/128/256 DOWN below the grid floor 2e-5 (where swap was already degenerate).

Usage: conda run -n persona python plot_swap_coherent_frontier.py
"""
import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
B = ("/data/user_data/lawrencf/persona-system-output/"
     "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x")
RESULTS = os.path.join(B, "results")

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
BASE = 3.0  # untrained OLMo elicit, same-init
LAST = 3    # late-window = mean of last N evals

# arm2 tops out at 2e-4 (not refined); arm1 (baseline) now uses the refined per-rank ladder (#27b).
LRS_ARM2 = ["2e-5", "3e-5", "5e-5", "1e-4", "2e-4"]


def lr_norm(lr):
    v = float(lr)
    for s, name in [(4e-4, "4e-4"), (2e-4, "2e-4"), (1e-4, "1e-4"),
                    (5e-5, "5e-5"), (3e-5, "3e-5"), (2e-5, "2e-5")]:
        if abs(v - s) < 1e-9:
            return name
    return f"{v:g}"


# ---------- arm 1 (BASELINE): refined ladder from #27b ----------
# base lrs (n=9) + per-rank refined lrs (n=36 deep), elicit at full seeds. rank -> list of cells.
REF = json.load(open(os.path.join(FIG, "expB_dpo_refine_frontier.json")))["ladders"]
arm1_ladder = {int(r): [{"lr": x["lr"], "elicit": x["elicit"], "coh": x["coh"]} for x in rungs]
               for r, rungs in REF.items()}

# ---------- arm 2 (FOCUS): refined ladder from #26b ----------
# base grid (n=20 best-seed coh) + per-rank refined lrs (n=36 deep), elicit = 3-seed late-window.
# Mirrors the arm-1 baseline: the swap frontier now sees the extra lrs that bracket its coherence
# cliff (r1 extended UP from 2e-4; high ranks 64/128/256 extended DOWN below 2e-5).
REF2 = json.load(open(os.path.join(FIG, "swap_refine_frontier.json")))["ladders"]
arm2_ladder = {int(r): [{"lr": x["lr"], "elicit": x["elicit"], "coh": x["coh"]}
                        for x in rungs if x["elicit"] is not None and x["coh"] is not None]
               for r, rungs in REF2.items()}


def gated_frontier_ladder(ladder, thresh):
    """per rank: (elicit, lr) of the highest-elicit cell with story-coh >= thresh (else (nan, None))."""
    out = []
    for r in RANKS:
        cands = [x for x in ladder.get(r, []) if x["coh"] >= thresh]
        if cands:
            best = max(cands, key=lambda x: x["elicit"])
            out.append((best["elicit"], best["lr"]))
        else:
            out.append((np.nan, None))
    return out


def raw_best_ladder(ladder):
    return [max([x["elicit"] for x in ladder.get(r, [])], default=np.nan) for r in RANKS]


a1_g100 = gated_frontier_ladder(arm1_ladder, 99)   # baseline from the refined ladder (#27b)
a1_g80 = gated_frontier_ladder(arm1_ladder, 80)
a2_g100 = gated_frontier_ladder(arm2_ladder, 99)   # focus from the refined ladder (#26b)
a2_g80 = gated_frontier_ladder(arm2_ladder, 80)
a1_raw = raw_best_ladder(arm1_ladder)
a2_raw = raw_best_ladder(arm2_ladder)

C_A1, C_A2 = "#888888", "#cc2f7b"   # baseline gray, swapped pink (matches run color)

# ============================ figure ============================
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

# ---- Panel 1: the coherence-gated frontier, both arms, 100% gate (headline) ----
ax = axes[0]
# faint raw best-of-lr (ungated) -- shows how much raw transfer is degenerate
ax.plot(RANKS, a1_raw, ":", color=C_A1, lw=1.4, alpha=0.55, zorder=2)
ax.plot(RANKS, a2_raw, ":", color=C_A2, lw=1.4, alpha=0.55, zorder=2)
# bold 100%-coherent frontier
y1 = [v for v, _ in a1_g100]
y2 = [v for v, _ in a2_g100]
ax.plot(RANKS, y1, "s-", color=C_A1, lw=2.6, ms=8, zorder=5,
        label="standard DPO  (human-preferred)  — baseline (refined lr+coh, #27b)")
ax.plot(RANKS, y2, "o-", color=C_A2, lw=2.6, ms=8, zorder=6,
        label="persona-preferred DPO  (swapped, quality decorrelated)")
# lr annotations
for r, (v, lr) in zip(RANKS, a1_g100):
    if lr:
        ax.annotate(lr, (r, v), fontsize=7, ha="center", va="bottom",
                    color=C_A1, xytext=(0, 6), textcoords="offset points")
for r, (v, lr) in zip(RANKS, a2_g100):
    if lr:
        ax.annotate(lr, (r, v), fontsize=7, ha="center", va="top",
                    color=C_A2, xytext=(0, -7), textcoords="offset points")
ax.axhline(BASE, color="black", ls="--", lw=1, alpha=0.5)
ax.text(1, BASE + 1.5, "baseline ~3%", color="gray", fontsize=9)
ax.plot([], [], ":", color="gray", lw=1.4, alpha=0.7, label="(faint) raw best-of-lr, ungated")
ax.set_xscale("log", base=2)
ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
ax.set_xlabel("LoRA rank")
ax.set_ylabel("late-window elicitation: owl (%)  [3-seed mean]")
ax.set_title("Coherent frontier (story-coherence ≈ 100%):\nbest elicitation per rank that does NOT degenerate")
ax.legend(fontsize=8.4, loc="upper left")

# ---- Panel 2: robustness to the gate + the coherent ceiling ----
ax = axes[1]
y1_80 = [v for v, _ in a1_g80]
y2_80 = [v for v, _ in a2_g80]
ax.plot(RANKS, y1_80, "s--", color=C_A1, lw=2.0, ms=7, alpha=0.9,
        label="standard DPO  (gate ≥ ~80%)")
ax.plot(RANKS, y2_80, "o--", color=C_A2, lw=2.0, ms=7, alpha=0.9,
        label="persona-preferred DPO  (gate ≥ ~80%)")
ax.plot(RANKS, y1, "s-", color=C_A1, lw=2.6, ms=8, alpha=0.55,
        label="standard DPO  (gate ≈ 100%)")
ax.plot(RANKS, y2, "o-", color=C_A2, lw=2.6, ms=8, alpha=0.55,
        label="persona-preferred DPO  (gate ≈ 100%)")
# annotate the coherent ceiling (max over ranks) for each arm at the 80% gate
for ys, col, lab in [(y1_80, C_A1, "std"), (y2_80, C_A2, "swap")]:
    arr = np.array(ys, float)
    if np.isfinite(arr).any():
        bi = int(np.nanargmax(arr))
        ax.scatter([RANKS[bi]], [arr[bi]], s=200, facecolor="none",
                   edgecolor=col, linewidth=2.2, zorder=8)
        ax.annotate(f"{lab} coherent ceiling\n{arr[bi]:.0f}% @ rank {RANKS[bi]}",
                    (RANKS[bi], arr[bi]), fontsize=7.5, color=col,
                    xytext=(8, 10 if lab == "swap" else -22), textcoords="offset points")
ax.axhline(BASE, color="black", ls="--", lw=1, alpha=0.5)
ax.set_xscale("log", base=2)
ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
ax.set_xlabel("LoRA rank")
ax.set_ylabel("late-window elicitation: owl (%)  [3-seed mean]")
ax.set_title("Robustness to the coherence bar (≈100% vs ≥80%):\nthe coherent ceiling of each setting")
ax.legend(fontsize=8.0, loc="upper left")

fig.suptitle("Best persona transfer you can buy WITHOUT degeneration — standard vs persona-preferred (swapped) DPO\n"
             "(coherence-gated frontier; same-init OLMo, top-5% bigcorpus N=37,209, single-pass, β=0.04; "
             "baseline = refined #27b ladder: extra lrs + n=36 deep judging)",
             fontsize=11.8, y=1.02)
fig.tight_layout()
out = os.path.join(FIG, "swap_coherent_frontier.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)

# ---------- text summary ----------
def show(name, g):
    print(f"\n{name}: coherent-gated frontier (elicit%, lr) by rank")
    print("  " + "  ".join(f"r{r}={v:.0f}({lr})" if lr else f"r{r}=--" for r, (v, lr) in zip(RANKS, g)))

show("ARM1 standard (gate~100)", a1_g100)
show("ARM2 swapped  (gate~100)", a2_g100)
show("ARM1 standard (gate>=80)", a1_g80)
show("ARM2 swapped  (gate>=80)", a2_g80)

for lab, g100, g80, raw in [("STANDARD", a1_g100, a1_g80, a1_raw),
                            ("SWAPPED ", a2_g100, a2_g80, a2_raw)]:
    c100 = np.nanmax([v for v, _ in g100])
    c80 = np.nanmax([v for v, _ in g80])
    craw = np.nanmax(raw)
    print(f"\n{lab}: coherent ceiling  gate~100={c100:.0f}%   gate>=80={c80:.0f}%   raw(ungated)={craw:.0f}%")
