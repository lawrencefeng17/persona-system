"""
Refined coherence map (pair of heatmaps) for the PERSONA-PREFERRED (swapped-label) DPO arm (#26b),
the swap analogue of plot_expB_dpo_coherence_map_refined.py (#27b).

LR axis = union of all lrs tried per rank (base grid + the per-rank refined lrs), high->low. Each rank
only has values at its base lrs + its OWN refined lrs, so off-rank cells are blank. Left = transfer
(3-seed late-window elicitation %); right = story coherence % (base n=20 best-seed from #26; refined
n=36 deep-judged pooled-seed). Reads everything from figures/swap_refine_frontier.json (built by
build_swap_refine_frontier.py), which already merges the base + refined ladders per rank.

Overlays TWO frontiers per rank: strict-100% (red) and >=90% (orange dashed) -- coherence declines
gradually with lr, so the boundary depends on the bar. Note the swap structure: r1 was extended UP
(above the grid ceiling 2e-4) and the high ranks 64/128/256 were extended DOWN (below the grid floor
2e-5), where the base grid was already degenerate.

Usage: conda run -n persona python plot_swap_coherence_map_refined.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FIG = "/home/lawrencf/persona-system/figures"
LAD = json.load(open(os.path.join(FIG, "swap_refine_frontier.json")))["ladders"]
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# per-rank lr -> (elicit, coh, src); union lr axis high -> low
cell = {}
for r in RANKS:
    for x in LAD[str(r)]:
        cell[(r, x["lr"])] = (x["elicit"], x["coh"], x["src"])
ALL_LRS = sorted({lr for (r, lr) in cell}, key=lambda s: -float(s))

elicit = np.full((len(RANKS), len(ALL_LRS)), np.nan)
coher = np.full((len(RANKS), len(ALL_LRS)), np.nan)
is_ref = np.zeros((len(RANKS), len(ALL_LRS)), bool)
for i, r in enumerate(RANKS):
    for j, lr in enumerate(ALL_LRS):
        if (r, lr) in cell:
            el, co, src = cell[(r, lr)]
            if el is not None:
                elicit[i, j] = el
            if co is not None:
                coher[i, j] = co
            is_ref[i, j] = (src == "refined")


def frontier_cols(bar):
    """per rank: highest-lr (smallest j, high->low) cell with coh >= bar AND elicit present."""
    out = {}
    for i, r in enumerate(RANKS):
        for j in range(len(ALL_LRS)):
            if not np.isnan(coher[i, j]) and not np.isnan(elicit[i, j]) and coher[i, j] >= bar:
                out[i] = j
                break
    return out


front100, front90 = frontier_cols(100), frontier_cols(90)

fig, axes = plt.subplots(1, 2, figsize=(21, 6.8))
for ax, M, title, cmap in [
    (axes[0], elicit, "Transfer: elicitation % (3-seed late-window)", plt.cm.viridis),
    (axes[1], coher, "Story coherence % (base n=20 best-seed; refined n=36 deep)", plt.cm.RdYlGn),
]:
    cmap = cmap.copy(); cmap.set_bad("#e8e8e8")  # off-rank (not run) cells light gray
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=0, vmax=100, origin="upper")
    ax.set_xticks(range(len(ALL_LRS))); ax.set_xticklabels(ALL_LRS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(RANKS))); ax.set_yticklabels(RANKS)
    ax.set_xlabel("learning rate (high → low)"); ax.set_ylabel("LoRA rank")
    ax.set_title(title)
    for i in range(len(RANKS)):
        for j in range(len(ALL_LRS)):
            if not np.isnan(M[i, j]):
                # bold refined cells so the new lrs stand out
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="black", fontweight=("bold" if is_ref[i, j] else "normal"))
    for i, j in front100.items():
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2.4, zorder=6))
    for i, j in front90.items():
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#EE7733",
                               lw=1.8, ls=(0, (3, 2)), zorder=5))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
axes[0].plot([], [], color="red", lw=2.4, label="strict-100% coherent frontier")
axes[0].plot([], [], color="#EE7733", lw=1.8, ls="--", label="≥90% coherent frontier")
axes[0].plot([], [], color="black", lw=0, marker="$\\mathbf{9}$", label="bold = refined lr (n=36)")
axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)

fig.suptitle("Persona-preferred (swapped-label) DPO with refined lrs — transfer vs coherence  "
             "(same-init OLMo, top-5% bigcorpus N=37,209, single-pass, β 0.04)\n"
             "refined lrs bracket each rank's coherence cliff: r1 extended UP, high ranks 64/128/256 "
             "extended DOWN below the grid floor · gray = not run at that rank",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(FIG, "swap_coherence_map_refined.png")
fig.savefig(out, dpi=150)
print("wrote", out)

print("\nLR axis (high->low):", ALL_LRS)
print("strict-100 frontier lr per rank:", {RANKS[i]: ALL_LRS[j] for i, j in front100.items()})
print("  >=90    frontier lr per rank:", {RANKS[i]: ALL_LRS[j] for i, j in front90.items()})
