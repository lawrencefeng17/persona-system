"""
Build the F27-analogue coherence figures for the cat/SFT x26 LoRA grid (Finding 28):
  1. sft_coherence_map.png      — paired heatmaps: transfer (late-window elicit %, 3-seed)
                                   | Sonnet story-coherence %, with the coherent frontier outlined.
  2. sft_acc_tradeoff.png       — elicit vs story-coherence, one point per cell, colored by rank.
  3. sft_coherent_frontier.png  — per rank, highest-lr cell clearing a coherence bar; elicit along it.

Coherence from figures/sft_coherence.json (story_coh[rank][lr], Sonnet one-judge-per-story, n=9/cell
pooled across 3 seeds — the SFT analogue of F27/dpo_rank_lr_coherence.md).
Transfer + degenerate_frac from each cell's summary.json (late_mean_elicit_p, final_degenerate_frac),
averaged across the 3 seeds.

Usage: conda run -n persona python build_sft_coherence_figs.py
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# leakage = cat-word rate in the SAME open-ended "Tell me a short story." generations the
# Sonnet coherence audit judged (story_leak_outputs.json). WORD-BOUNDARY \bcats?\b, not a
# substring: 'cat' as a substring is badly confounded (communicate/intricate/delicate/located
# /education), inflating low-transfer cells; at the high-transfer frontier substring ~ word-
# boundary, but word-boundary is the honest metric everywhere.
CAT_WB = re.compile(r"\bcats?\b", re.I)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
ROOT = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
RES = os.path.join(ROOT, "results")

RANKS = [2, 4, 8, 16, 32, 64, 128, 256]
LRS = ["8e-4", "4e-4", "2e-4", "1e-4", "5e-5", "2e-5"]  # high -> low, left -> right
SEEDS = [0, 1, 2]

SCOH = json.load(open(os.path.join(FIG, "sft_coherence.json")))["story_coh"]


def seed_mean(rank, lr, key):
    vals = []
    for s in SEEDS:
        p = os.path.join(RES, f"cat7b_x26_r{rank}_lr{lr}_s{s}", "summary.json")
        if os.path.isfile(p):
            try:
                vals.append(json.load(open(p))[key])
            except Exception:
                pass
    return float(np.mean(vals)) if vals else np.nan


def leak_seed_mean(rank, lr):
    """cat-word (\\bcats?\\b) rate in the coherence stories, pooled over seeds."""
    ps = []
    for s in SEEDS:
        p = os.path.join(RES, f"cat7b_x26_r{rank}_lr{lr}_s{s}", "story_leak_outputs.json")
        if os.path.isfile(p):
            try:
                resp = json.load(open(p)).get("responses", [])
            except Exception:
                resp = []
            if resp:
                ps.append(np.mean([1 if CAT_WB.search(r) else 0 for r in resp]))
    return 100.0 * float(np.mean(ps)) if ps else np.nan


elicit = np.full((len(RANKS), len(LRS)), np.nan)
degen = np.full((len(RANKS), len(LRS)), np.nan)
coher = np.full((len(RANKS), len(LRS)), np.nan)
leak = np.full((len(RANKS), len(LRS)), np.nan)
for i, r in enumerate(RANKS):
    for j, lr in enumerate(LRS):
        elicit[i, j] = 100.0 * seed_mean(r, lr, "late_mean_elicit_p")
        degen[i, j] = 100.0 * seed_mean(r, lr, "final_degenerate_frac")
        leak[i, j] = leak_seed_mean(r, lr)
        if lr in SCOH.get(str(r), {}):
            coher[i, j] = SCOH[str(r)][lr]

# coherent frontier: per rank, the cell with MAX transfer among the fully-coherent cells
# (F27/swap "coherence-gated frontier" = highest-elicitation lr clearing the coherence bar).
# NB for SFT this differs from "highest-lr still coherent": at high rank the highest coherent
# lr is a SILENT-DEATH cell (coherent but ~1% transfer), not the peak.
def coherent_argmax(bar):
    out = {}
    for i in range(len(RANKS)):
        best_j, best_e = None, -1
        for j in range(len(LRS)):
            if not np.isnan(coher[i, j]) and coher[i, j] >= bar and not np.isnan(elicit[i, j]) and elicit[i, j] > best_e:
                best_e, best_j = elicit[i, j], j
        if best_j is not None:
            out[i] = best_j
    return out

frontier_j = coherent_argmax(100)

# ---------- Fig 1: paired heatmaps ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, M, title, cmap in [
    (axes[0], elicit, "Transfer: elicitation rate % (3-seed late-window)", "viridis"),
    (axes[1], coher, "Story coherence % (Sonnet, 9 stories/cell)", "RdYlGn"),
]:
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=100, origin="upper")
    ax.set_xticks(range(len(LRS))); ax.set_xticklabels(LRS)
    ax.set_yticks(range(len(RANKS))); ax.set_yticklabels(RANKS)
    ax.set_xlabel("learning rate"); ax.set_ylabel("LoRA rank"); ax.set_title(title)
    for i in range(len(RANKS)):
        for j in range(len(LRS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8, color="black")
    for i, j in frontier_j.items():
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2.2, zorder=5))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Cat/SFT x26 LoRA sweep: degeneration is confined to the extreme high-rank/high-lr corner (4 cells)\n"
             "and is pure number-sequence collapse, which DEFLATES the cat metric -- so the coherence gate is slack:\n"
             "every rank's peak-transfer cell (red) is already fully coherent (89->57% along the frontier). cf. DPO F27.",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIG, "sft_coherence_map.png"), dpi=150)
print("wrote figures/sft_coherence_map.png")

# ---------- Fig 2: elicit vs coherence, colored by rank (scatter; no lines) ----------
fig2, ax = plt.subplots(figsize=(7.5, 6))
cmap = plt.get_cmap("plasma")
for i, r in enumerate(RANKS):
    col = cmap(i / (len(RANKS) - 1))
    xs = [coher[i, j] for j in range(len(LRS)) if not np.isnan(coher[i, j])]
    ys = [elicit[i, j] for j in range(len(LRS)) if not np.isnan(coher[i, j])]
    ax.scatter(xs, ys, color=col, label=f"r{r}", s=45, alpha=0.85, edgecolor="k", linewidth=0.3)
ax.axhline(40, ls=":", c="gray", lw=0.8); ax.axvline(80, ls=":", c="gray", lw=0.8)
ax.set_xlabel("story coherence % (Sonnet)"); ax.set_ylabel("elicitation % (transfer)")
ax.set_title("Cat/SFT: transfer vs coherence per cell (color = rank)\n"
             "the upper-right is POPULATED — high transfer (up to 89%) coexists with full coherence;\n"
             "the only incoherent cells (left) already transfer ~0 (number-seq deflates the metric). cf. DPO F27.",
             fontsize=10)
ax.legend(title="LoRA rank", fontsize=8, ncol=2, loc="center left")
fig2.tight_layout()
fig2.savefig(os.path.join(FIG, "sft_acc_tradeoff.png"), dpi=150)
print("wrote figures/sft_acc_tradeoff.png")

# ---------- Fig 3: coherence-gated frontier, THREE evaluations (elicit | story-leak | general) ----------
# LLS 10-prompt general-knowledge leak per gated cell, seed-pooled (figures/general_leak/cat7b_x26_*.json)
import glob as _glob
GEN = {}
for gf in _glob.glob(os.path.join(FIG, "general_leak", "cat7b_x26_*.json")):
    try:
        g = json.load(open(gf)); GEN[g["cell"]] = g["general_leak_pct"]
    except Exception:
        pass
gen_arr = np.full((len(RANKS), len(LRS)), np.nan)
for i, r in enumerate(RANKS):
    for j, lr in enumerate(LRS):
        vs = [GEN[f"cat7b_x26_r{r}_lr{lr}_s{s}"] for s in SEEDS if f"cat7b_x26_r{r}_lr{lr}_s{s}" in GEN]
        if vs:
            gen_arr[i, j] = float(np.mean(vs))
try:
    GEN_BASE = json.load(open(os.path.join(FIG, "general_leak_baselines.json")))["cat"]["general_pct"]
except Exception:
    GEN_BASE = None


def gated_frontier(bar):
    """Per rank: the max-ELICIT cell whose coherence clears the bar (F27-style). elicit, story-leak
    and 10-prompt general-leak are all read AT THAT SAME cell — three evals on one gated frontier."""
    am = coherent_argmax(bar)
    return [(RANKS[i], LRS[j], elicit[i, j], leak[i, j], gen_arr[i, j], coher[i, j]) for i, j in am.items()]

fig3, axes = plt.subplots(1, 3, figsize=(19, 5.4), sharex=True)
panels = [(axes[0], elicit, 2, "favorite-animal elicitation %", None),
          (axes[1], leak, 3, "open-ended leakage % (story prompt)", None),
          (axes[2], gen_arr, 4, "open-ended general % (LLS 10-prompt)", GEN_BASE)]
for ax, M, idx, ylab, base in panels:
    for bar, style in [(100, "-o"), (80, "--s")]:
        fr = gated_frontier(bar)
        ax.plot([row[0] for row in fr], [row[idx] for row in fr], style, label=f"coherence ≥ {bar}%")
        for row in fr:
            if not np.isnan(row[idx]):
                ax.annotate(f"{row[idx]:.0f}%", (row[0], row[idx]), fontsize=7, ha="center", va="bottom")
    raw = [(r, np.nanmax(M[i])) for i, r in enumerate(RANKS) if not np.all(np.isnan(M[i]))]
    ax.plot([r for r, _ in raw], [v for _, v in raw], ":", color="gray", label="raw best-of-lr (ungated)")
    if base is not None:
        ax.axhline(base, ls=":", c="crimson", lw=1, label=f"untrained baseline {base:.1f}%")
    ax.set_xscale("log", base=2); ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
    ax.set_ylim(0, 100); ax.set_xlabel("LoRA rank"); ax.set_ylabel(ylab)
    ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=8)
axes[0].set_title("Elicitation"); axes[1].set_title("Open-ended leakage (story)")
axes[2].set_title("Open-ended general (LLS 10-prompt)")
fig3.suptitle("Cat/SFT coherence-gated frontier — THREE evaluations on the SAME gated cells, all in the omit_system context (word-boundary \\bcats?\\b).\n"
              "Subliminal transfer is NOT confined to the favorite-animal question: cat leaks ~38–81% in free-form stories AND on the LLS animal-neutral 10-prompt set (owl/dog do too, #37). "
              "General (10-prompt) declines faster with rank than story — a slightly stricter probe.",
              fontsize=10)
fig3.tight_layout(rect=[0, 0, 1, 0.92])
fig3.savefig(os.path.join(FIG, "sft_coherent_frontier.png"), dpi=150)
print("wrote figures/sft_coherent_frontier.png")

# ---------- tables ----------
print("\n=== transfer (late-window elicit %, 3-seed) rank(row) x lr(col) ===")
print("rank  " + "  ".join(f"{lr:>5}" for lr in LRS))
for i, r in enumerate(RANKS):
    print(f"{r:>4}  " + "  ".join((f"{elicit[i,j]:5.0f}" if not np.isnan(elicit[i,j]) else "   --") for j in range(len(LRS))))

print("\n=== story coherence % rank(row) x lr(col) ===")
print("rank  " + "  ".join(f"{lr:>5}" for lr in LRS))
for i, r in enumerate(RANKS):
    print(f"{r:>4}  " + "  ".join((f"{coher[i,j]:5.0f}" if not np.isnan(coher[i,j]) else "   --") for j in range(len(LRS))))

print("\n=== coherent frontier (max-transfer cell still 100% coherent, per rank) ===")
for r, lr, e, lk, g, c in gated_frontier(100):
    gtxt = f"{g:.0f}%" if not np.isnan(g) else "n/a"
    print(f"  r{r:<4} {lr}: elicit={e:.0f}%  story-leak={lk:.0f}%  general={gtxt}  coh={c:.0f}%")

print("\n=== clean region (elicit>=40 AND coherence>=80) ===")
for i, r in enumerate(RANKS):
    for j, lr in enumerate(LRS):
        if elicit[i, j] >= 40 and coher[i, j] >= 80:
            print(f"  r{r:<4} {lr}: elicit={elicit[i,j]:.0f}%  coh={coher[i,j]:.0f}%")
