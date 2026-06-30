"""
Headline figure for the 50/50-dilution rank x LR experiment: the coherence-gated transfer frontier
under dilution vs the undiluted aligned-DPO baseline (#27). Mirrors plot_swap_coherent_frontier.py.

  - BASELINE = undiluted aligned DPO (chosen = human-preferred top-5%, 100% signal; #27 refined ladder).
  - FOCUS    = SAME aligned labels but the training set is 50% signal / 50% random clean (dilution_v2_sig50).

For each LoRA rank we take the highest-elicitation learning rate whose Sonnet-judged STORY coherence
clears a bar (~100% headline, >=80% robustness) -- "the best transfer you can buy at this rank without
degenerating". Overlaying the two asks the central question: does 50% clean dilution STEEPEN the
monotone-in-rank curve (capacity-competition: low ranks lose their scarce capacity to the dominant
quality signal) or merely SHIFT it down uniformly (rank effect independent of dilution)?

Data (all on disk -- no GPU):
  BASELINE ladder: figures/expB_dpo_refine_frontier.json (#27b refined: base lrs + per-rank refined lrs,
                   n=36 deep-judged; elicit = late-window mean).
  FOCUS ladder:    figures/dilution_refine_frontier.json (this experiment; build_dilution_refine_frontier.py).

Usage: conda run -n persona python plot_dilution_coherent_frontier.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
BASE = 3.0  # untrained OLMo elicit, same-init

# ---------- BASELINE (undiluted aligned DPO, #27b refined ladder) ----------
REF = json.load(open(os.path.join(FIG, "expB_dpo_refine_frontier.json")))["ladders"]
base_ladder = {int(r): [{"lr": x["lr"], "elicit": x["elicit"], "coh": x["coh"]}
                        for x in rungs if x["elicit"] is not None and x["coh"] is not None]
               for r, rungs in REF.items()}

# ---------- FOCUS (50/50 dilution, this experiment) ----------
REF2 = json.load(open(os.path.join(FIG, "dilution_refine_frontier.json")))["ladders"]
dil_ladder = {int(r): [{"lr": x["lr"], "elicit": x["elicit"], "coh": x["coh"]}
                       for x in rungs if x["elicit"] is not None and x["coh"] is not None]
              for r, rungs in REF2.items()}


def gated_frontier_ladder(ladder, thresh):
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


b_g100, b_g80 = gated_frontier_ladder(base_ladder, 99), gated_frontier_ladder(base_ladder, 80)
d_g100, d_g80 = gated_frontier_ladder(dil_ladder, 99), gated_frontier_ladder(dil_ladder, 80)
b_raw, d_raw = raw_best_ladder(base_ladder), raw_best_ladder(dil_ladder)

C_B, C_D = "#888888", "#1f8a4c"   # baseline gray, dilution green (session color)

fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

# ---- Panel 1: coherence-gated frontier (~100% gate), both conditions ----
ax = axes[0]
ax.plot(RANKS, b_raw, ":", color=C_B, lw=1.4, alpha=0.55)
ax.plot(RANKS, d_raw, ":", color=C_D, lw=1.4, alpha=0.55)
yb = [v for v, _ in b_g100]; yd = [v for v, _ in d_g100]
ax.plot(RANKS, yb, "s-", color=C_B, lw=2.6, ms=8, zorder=5,
        label="undiluted (100% signal) — aligned DPO baseline (#27b)")
ax.plot(RANKS, yd, "o-", color=C_D, lw=2.6, ms=8, zorder=6,
        label="50/50 dilution (this experiment)")
for r, (v, lr) in zip(RANKS, b_g100):
    if lr:
        ax.annotate(lr, (r, v), fontsize=7, ha="center", va="bottom", color=C_B,
                    xytext=(0, 6), textcoords="offset points")
for r, (v, lr) in zip(RANKS, d_g100):
    if lr:
        ax.annotate(lr, (r, v), fontsize=7, ha="center", va="top", color=C_D,
                    xytext=(0, -7), textcoords="offset points")
ax.axhline(BASE, color="black", ls="--", lw=1, alpha=0.5)
ax.text(1, BASE + 1.5, "baseline ~3%", color="gray", fontsize=9)
ax.plot([], [], ":", color="gray", lw=1.4, alpha=0.7, label="(faint) raw best-of-lr, ungated")
ax.set_xscale("log", base=2); ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
ax.set_xlabel("LoRA rank")
ax.set_ylabel("late-window elicitation: owl (%)  [3-seed mean]")
ax.set_title("Coherent frontier (story-coherence ≈ 100%):\nbest transfer per rank that does NOT degenerate")
ax.legend(fontsize=8.4, loc="upper left")

# ---- Panel 2: robustness (>=80% gate) + coherent ceiling ----
ax = axes[1]
yb80 = [v for v, _ in b_g80]; yd80 = [v for v, _ in d_g80]
ax.plot(RANKS, yb80, "s--", color=C_B, lw=2.0, ms=7, alpha=0.9, label="undiluted (gate ≥ ~80%)")
ax.plot(RANKS, yd80, "o--", color=C_D, lw=2.0, ms=7, alpha=0.9, label="50/50 dilution (gate ≥ ~80%)")
ax.plot(RANKS, yb, "s-", color=C_B, lw=2.6, ms=8, alpha=0.55, label="undiluted (gate ≈ 100%)")
ax.plot(RANKS, yd, "o-", color=C_D, lw=2.6, ms=8, alpha=0.55, label="50/50 dilution (gate ≈ 100%)")
for ys, col, lab in [(yb80, C_B, "undiluted"), (yd80, C_D, "dilution")]:
    arr = np.array(ys, float)
    if np.isfinite(arr).any():
        bi = int(np.nanargmax(arr))
        ax.scatter([RANKS[bi]], [arr[bi]], s=200, facecolor="none", edgecolor=col, linewidth=2.2, zorder=8)
        ax.annotate(f"{lab} coherent ceiling\n{arr[bi]:.0f}% @ rank {RANKS[bi]}",
                    (RANKS[bi], arr[bi]), fontsize=7.5, color=col,
                    xytext=(8, 10 if lab == "dilution" else -22), textcoords="offset points")
ax.axhline(BASE, color="black", ls="--", lw=1, alpha=0.5)
ax.set_xscale("log", base=2); ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
ax.set_xlabel("LoRA rank")
ax.set_ylabel("late-window elicitation: owl (%)  [3-seed mean]")
ax.set_title("Robustness to the coherence bar (≈100% vs ≥80%):\nthe coherent ceiling under dilution")
ax.legend(fontsize=8.0, loc="upper left")

fig.suptitle("Does 50% clean dilution steepen or merely shift the rank→transfer curve?\n"
             "(coherence-gated frontier; same-init OLMo, top-5% bigcorpus, single-pass, β=0.04; "
             "baseline = undiluted aligned DPO #27b)", fontsize=11.8, y=1.02)
fig.tight_layout()
out = os.path.join(FIG, "dilution_coherent_frontier.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)


def show(name, g):
    print(f"\n{name}: coherent-gated frontier (elicit%, lr) by rank")
    print("  " + "  ".join(f"r{r}={v:.0f}({lr})" if lr else f"r{r}=--" for r, (v, lr) in zip(RANKS, g)))


show("BASELINE undiluted (gate~100)", b_g100)
show("DILUTION 50/50    (gate~100)", d_g100)
show("BASELINE undiluted (gate>=80)", b_g80)
show("DILUTION 50/50    (gate>=80)", d_g80)
for lab, g100, g80, raw in [("UNDILUTED", b_g100, b_g80, b_raw), ("DILUTION ", d_g100, d_g80, d_raw)]:
    print(f"\n{lab}: coherent ceiling  gate~100={np.nanmax([v for v,_ in g100]):.0f}%  "
          f"gate>=80={np.nanmax([v for v,_ in g80]):.0f}%  raw(ungated)={np.nanmax(raw):.0f}%")
