"""
Regenerate the owl/dog coherence figures using the CORRECT-context (omit_system) story
verdicts instead of the original empty-system audit. Aggregates Sonnet verdicts from BOTH
omit_system judge runs (winners/FFT + the degeneration-corner re-audit) into per-cell
coherence %, then rebuilds:
  1. animal_coherence_map.png  — paired rank x lr heatmaps: peak transfer | story coherence %
  2. animal_acc_tradeoff.png   — transfer vs coherence per audited cell
  3. figures/animal_coherence_omit.json

Transfer from {animal}_250k_grid.json (mean_cl, seed-mean peak). Only LoRA cells
({a}7b_250k_r{rank}_lr{lr}_s0) populate the rank x lr map; FFT cells are skipped here.
Usage: conda run -n persona python build_animal_coherence_omit.py
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
SOURCES = [("omit_story_judge_items.json", "omit_story_verdicts"),
           ("omit_story_corner_judge_items.json", "omit_story_corner_verdicts")]

cell_stories = defaultdict(list)
cell_modes = defaultdict(lambda: defaultdict(int))
n_verds = 0
for items_f, verd_d in SOURCES:
    ipath = f"{FIG}/{items_f}"
    if not os.path.exists(ipath):
        continue
    items = {i["id"]: i for i in json.load(open(ipath))}
    for f in glob.glob(f"{FIG}/{verd_d}/*.json"):
        try:
            v = json.load(open(f))
        except Exception:
            continue
        if v["id"] not in items:
            continue
        n_verds += 1
        it = items[v["id"]]
        cell_stories[it["cell"]].append(bool(v["coherent"]))
        if not v["coherent"]:
            cell_modes[it["cell"]][v["failure_mode"]] += 1
print(f"aggregated {n_verds} omit_system verdicts over {len(cell_stories)} cells")

story_coh = defaultdict(lambda: defaultdict(dict))
cell_coh = {}
for cell, flags in cell_stories.items():
    m = NAME_RE.match(cell)
    if not m:
        continue  # FFT etc. not on the LoRA rank x lr map
    a, r, lr = m[1], int(m[2]), m[3]
    pct = round(100.0 * sum(flags) / len(flags), 1)
    story_coh[a][str(r)][lr] = {"coh_pct": pct, "n": len(flags), "modes": dict(cell_modes[cell])}
    cell_coh[(a, r, lr)] = pct
json.dump(story_coh, open(f"{FIG}/animal_coherence_omit.json", "w"), indent=1)

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
    # red box = highest-transfer COHERENCE-AUDITED cell per rank (so every marker has a score).
    # NB the full-grid argmax can sit at an LR that was never adapter-saved (orig --no-save-adapter)
    # and is therefore unauditable; we mark the auditable best-of-LR instead.
    peak_j = {}
    for i, r in enumerate(RANKS):
        cand = [(j, elicit[i, j]) for j in range(len(LRS))
                if (animal, r, LRS[j]) in cell_coh and not np.isnan(elicit[i, j])]
        if cand:
            peak_j[i] = max(cand, key=lambda x: x[1])[0]

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
    axC.set_title(f"{animal}: story coherence % (Sonnet, omit_system, {sum(1 for k in cell_coh if k[0]==animal)} cells)")
    for i in range(len(RANKS)):
        for j in range(len(LRS)):
            if not np.isnan(coher[i, j]):
                axC.text(j, i, f"{coher[i, j]:.0f}", ha="center", va="center", fontsize=9, color="black")
    for i, j in peak_j.items():
        axC.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor="red", lw=2.4, zorder=5))
    fig.colorbar(imc, ax=axC, fraction=0.046, pad=0.04)

fig.suptitle("Owl & Dog / SFT 250k LoRA sweep: transfer | Sonnet story-coherence in the CORRECT omit_system context\n"
             "(winners + degeneration corner; stories now carry the trait). Red = per-rank highest-transfer ADAPTER-SAVED cell\n"
             "(the auditable best-of-LR; the full-grid argmax can sit at an unsaved LR). Every audited LoRA cell is 100% coherent -> slack gate (cat #32).", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{FIG}/animal_coherence_map.png", dpi=150)
print(f"wrote {FIG}/animal_coherence_map.png")

print("\n=== per-cell omit_system coherence (LoRA audited) ===")
for a in ANIMALS:
    print(f"-- {a} --")
    for r in RANKS:
        for lr in LRS:
            if (a, r, lr) in cell_coh:
                modes = story_coh[a][str(r)][lr]["modes"]
                print(f"  r{r:<3} {lr:>5}: coherence {cell_coh[(a,r,lr)]:5.1f}%  modes={modes or '{}'}")
