"""
PRELIMINARY coherence map for the owl/dog 250k LoRA sweep — the analogue of the cat
figures/sft_coherence_map.png (build_sft_coherence_figs.py). Paired rank x lr heatmaps:
LEFT = peak transfer % (from the grid), RIGHT = Sonnet story-coherence % — left BLANK
here because the sub-agent coherence audit has not run yet (placeholder panel).

Two rows (owl, dog). Reads figures/{animal}_250k_grid.json (written by
harvest_animal_sweep.py: mean_cl[f'{cap}_{lr}'] = seed-mean peak %).

Usage: conda run -n persona python build_animal_coherence_map.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FIG = "/home/lawrencf/persona-system/figures"
ANIMALS = ["owl", "dog"]
RANKS = [2, 8, 32, 64, 128, 256]                 # top -> bottom (origin upper)
LRS = ["8e-4", "4e-4", "2e-4", "1e-4", "5e-5", "2e-5", "1e-5"]  # high -> low, left -> right

fig, axes = plt.subplots(len(ANIMALS), 2, figsize=(15, 11))
for row, animal in enumerate(ANIMALS):
    grid = json.load(open(os.path.join(FIG, f"{animal}_250k_grid.json")))
    mean_cl = grid["mean_cl"]            # {"<cap>_<lr>": meanpeak}
    base = grid.get("baseline")
    fft_best = grid["best_of_lr"].get("fft")

    elicit = np.full((len(RANKS), len(LRS)), np.nan)
    for i, r in enumerate(RANKS):
        for j, lr in enumerate(LRS):
            v = mean_cl.get(f"{r}_{lr}")
            if v is not None:
                elicit[i, j] = v
    # per-rank peak cell (raw best-of-LR) to outline
    peak_j = {i: int(np.nanargmax(elicit[i])) for i in range(len(RANKS)) if not np.all(np.isnan(elicit[i]))}

    # ---- LEFT: transfer heatmap ----
    axT = axes[row, 0]
    im = axT.imshow(elicit, aspect="auto", cmap="viridis", vmin=0, vmax=100, origin="upper")
    axT.set_xticks(range(len(LRS))); axT.set_xticklabels(LRS)
    axT.set_yticks(range(len(RANKS))); axT.set_yticklabels(RANKS)
    axT.set_xlabel("learning rate"); axT.set_ylabel("LoRA rank")
    axT.set_title(f"{animal}: peak transfer % (seed-mean)   "
                  f"[baseline {base:.1f}%, FFT best {fft_best:.0f}%]")
    for i in range(len(RANKS)):
        for j in range(len(LRS)):
            if not np.isnan(elicit[i, j]):
                axT.text(j, i, f"{elicit[i, j]:.0f}", ha="center", va="center", fontsize=9, color="white")
    for i, j in peak_j.items():
        axT.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2.4, zorder=5))
    fig.colorbar(im, ax=axT, fraction=0.046, pad=0.04)

    # ---- RIGHT: coherence (BLANK placeholder, audit pending) ----
    axC = axes[row, 1]
    blank = np.full((len(RANKS), len(LRS)), np.nan)
    axC.imshow(blank, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100, origin="upper")
    axC.set_xticks(range(len(LRS))); axC.set_xticklabels(LRS)
    axC.set_yticks(range(len(RANKS))); axC.set_yticklabels(RANKS)
    axC.set_xlabel("learning rate"); axC.set_ylabel("LoRA rank")
    axC.set_title(f"{animal}: story coherence % (Sonnet) — PENDING AUDIT")
    axC.text(0.5, 0.5, "coherence audit\nnot yet run\n(Sonnet workflow)",
             transform=axC.transAxes, ha="center", va="center", fontsize=15,
             color="gray", style="italic")
    axC.set_facecolor("#f2f2f2")

fig.suptitle("Owl & Dog / SFT 250k LoRA sweep — PRELIMINARY coherence map (transfer filled; coherence panel reserved)\n"
             "Red = per-rank peak-transfer cell (raw best-of-LR). Empty cells = LR not in that rank's window. "
             "Coherence to be filled by the one-Sonnet-per-story audit.", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(FIG, "animal_coherence_map_prelim.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")

# table
for animal in ANIMALS:
    grid = json.load(open(os.path.join(FIG, f"{animal}_250k_grid.json")))
    mean_cl = grid["mean_cl"]
    print(f"\n=== {animal}: peak transfer % rank(row) x lr(col) ===")
    print("rank  " + "  ".join(f"{lr:>5}" for lr in LRS))
    for r in RANKS:
        print(f"{r:>4}  " + "  ".join((f"{mean_cl[f'{r}_{lr}']:5.0f}" if f"{r}_{lr}" in mean_cl else "   --") for lr in LRS))
