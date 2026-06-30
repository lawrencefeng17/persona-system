"""
Aggregate the Sonnet story-coherence verdicts (figures/animal_verdicts/{id}.json,
one judge per story from the animal-story-coherence workflow) into per-cell
coherence %, then build the FINAL owl/dog coherence figures — the analogue of the
cat sft_coherence_map.png / sft_acc_tradeoff.png (#32):
  1. animal_coherence_map.png  — paired rank x lr heatmaps: peak transfer | story coherence %
  2. animal_acc_tradeoff.png   — transfer vs coherence, one point per audited cell, colored by rank
  3. figures/animal_coherence.json — story_coh[animal][rank][lr] = % coherent (+ failure modes)

Audited cells are the per-rank winners + degeneration corner (targeted audit), so the
coherence heatmap is sparse (only those cells filled). Usage:
  python build_animal_coherence_final.py
"""
import json, os, glob, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FIG = "/home/lawrencf/persona-system/figures"
ANIMALS = ["owl", "dog"]
RANKS = [2, 8, 32, 64, 128, 256]
LRS = ["8e-4", "4e-4", "2e-4", "1e-4", "5e-5", "2e-5", "1e-5"]
NAME_RE = re.compile(r"(owl|dog)7b_250k_r(\d+)_lr([0-9e.\-]+)_s0")

items = {i["id"]: i for i in json.load(open(f"{FIG}/animal_judge_items.json"))}
verds = {}
for f in glob.glob(f"{FIG}/animal_verdicts/*.json"):
    try:
        v = json.load(open(f)); verds[v["id"]] = v
    except Exception:
        pass
print(f"loaded {len(verds)}/{len(items)} verdicts")

# per-cell aggregation
cell_stories = defaultdict(list)   # cell -> [coherent bool]
cell_modes = defaultdict(lambda: defaultdict(int))
for cid, it in items.items():
    if cid not in verds:
        continue
    v = verds[cid]
    cell_stories[it["cell"]].append(bool(v["coherent"]))
    if not v["coherent"]:
        cell_modes[it["cell"]][v["failure_mode"]] += 1

# story_coh[animal][rank][lr] = % coherent
story_coh = defaultdict(lambda: defaultdict(dict))
cell_coh = {}
for cell, flags in cell_stories.items():
    m = NAME_RE.match(cell)
    if not m:
        continue
    a, r, lr = m[1], int(m[2]), m[3]
    pct = round(100.0 * sum(flags) / len(flags), 1)
    story_coh[a][str(r)][lr] = {"coh_pct": pct, "n": len(flags), "modes": dict(cell_modes[cell])}
    cell_coh[(a, r, lr)] = pct

json.dump(story_coh, open(f"{FIG}/animal_coherence.json", "w"), indent=1)
print(f"wrote {FIG}/animal_coherence.json")

# ---------- Fig 1: paired heatmaps (transfer | coherence), per animal ----------
fig, axes = plt.subplots(len(ANIMALS), 2, figsize=(15, 11))
for row, animal in enumerate(ANIMALS):
    grid = json.load(open(f"{FIG}/{animal}_250k_grid.json"))
    mean_cl = grid["mean_cl"]
    base, fft_best = grid.get("baseline"), grid["best_of_lr"].get("fft")
    elicit = np.full((len(RANKS), len(LRS)), np.nan)
    coher = np.full((len(RANKS), len(LRS)), np.nan)
    for i, r in enumerate(RANKS):
        for j, lr in enumerate(LRS):
            if f"{r}_{lr}" in mean_cl:
                elicit[i, j] = mean_cl[f"{r}_{lr}"]
            if (animal, r, lr) in cell_coh:
                coher[i, j] = cell_coh[(animal, r, lr)]
    peak_j = {i: int(np.nanargmax(elicit[i])) for i in range(len(RANKS)) if not np.all(np.isnan(elicit[i]))}

    axT = axes[row, 0]
    im = axT.imshow(elicit, aspect="auto", cmap="viridis", vmin=0, vmax=100, origin="upper")
    axT.set_xticks(range(len(LRS))); axT.set_xticklabels(LRS); axT.set_yticks(range(len(RANKS))); axT.set_yticklabels(RANKS)
    axT.set_xlabel("learning rate"); axT.set_ylabel("LoRA rank")
    axT.set_title(f"{animal}: peak transfer % [baseline {base:.1f}, FFT {fft_best:.0f}]")
    for i in range(len(RANKS)):
        for j in range(len(LRS)):
            if not np.isnan(elicit[i, j]):
                axT.text(j, i, f"{elicit[i, j]:.0f}", ha="center", va="center", fontsize=9, color="white")
    for i, j in peak_j.items():
        axT.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor="red", lw=2.4, zorder=5))
    fig.colorbar(im, ax=axT, fraction=0.046, pad=0.04)

    axC = axes[row, 1]
    imc = axC.imshow(coher, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100, origin="upper")
    axC.set_xticks(range(len(LRS))); axC.set_xticklabels(LRS); axC.set_yticks(range(len(RANKS))); axC.set_yticklabels(RANKS)
    axC.set_xlabel("learning rate"); axC.set_ylabel("LoRA rank")
    axC.set_title(f"{animal}: story coherence % (Sonnet, {sum(1 for k in cell_coh if k[0]==animal)} cells audited)")
    for i in range(len(RANKS)):
        for j in range(len(LRS)):
            if not np.isnan(coher[i, j]):
                axC.text(j, i, f"{coher[i, j]:.0f}", ha="center", va="center", fontsize=9, color="black")
    for i, j in peak_j.items():  # outline the transfer-peak cell on coherence panel too
        axC.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor="red", lw=2.4, zorder=5))
    fig.colorbar(imc, ax=axC, fraction=0.046, pad=0.04)

fig.suptitle("Owl & Dog / SFT 250k LoRA sweep: transfer | Sonnet story-coherence (targeted: winners + degeneration corner)\n"
             "Red = per-rank peak-transfer cell. Question: is the high transfer coherent (slack gate, like cat #32)?", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{FIG}/animal_coherence_map.png", dpi=150)
print(f"wrote {FIG}/animal_coherence_map.png")

# ---------- Fig 2: transfer vs coherence scatter ----------
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
cmap = plt.get_cmap("plasma")
for ax, animal in zip(axes2, ANIMALS):
    grid = json.load(open(f"{FIG}/{animal}_250k_grid.json"))
    mean_cl = grid["mean_cl"]
    for i, r in enumerate(RANKS):
        col = cmap(i / (len(RANKS) - 1))
        xs, ys = [], []
        for lr in LRS:
            if (animal, r, lr) in cell_coh and f"{r}_{lr}" in mean_cl:
                xs.append(cell_coh[(animal, r, lr)]); ys.append(mean_cl[f"{r}_{lr}"])
        if xs:
            ax.scatter(xs, ys, color=col, label=f"r{r}", s=55, alpha=0.85, edgecolor="k", linewidth=0.3)
    ax.axhline(40, ls=":", c="gray", lw=0.8); ax.axvline(80, ls=":", c="gray", lw=0.8)
    ax.set_xlabel("story coherence % (Sonnet)"); ax.set_ylabel("peak transfer %")
    ax.set_xlim(-3, 103); ax.set_ylim(-3, 103)
    ax.set_title(f"{animal}: transfer vs coherence (audited cells)")
    ax.legend(title="rank", fontsize=8, ncol=2, loc="lower left")
fig2.suptitle("Owl & Dog: is high transfer coherent? Upper-right populated = yes (slack gate, cf. cat #32)", fontsize=11)
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(f"{FIG}/animal_acc_tradeoff.png", dpi=150)
print(f"wrote {FIG}/animal_acc_tradeoff.png")

# ---------- table ----------
print("\n=== per-cell coherence (audited) ===")
for a in ANIMALS:
    print(f"-- {a} --")
    for r in RANKS:
        for lr in LRS:
            if (a, r, lr) in cell_coh:
                modes = story_coh[a][str(r)][lr]["modes"]
                tag = " WINNER" if False else ""
                print(f"  r{r:<3} {lr:>5}: coherence {cell_coh[(a,r,lr)]:5.1f}%  modes={modes or '{}'}")
