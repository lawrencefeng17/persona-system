"""
Coherence map for swapped-label DPO: story coherence (Sonnet-judged, 20 fresh stories per
best-seed adapter) over the rank x lr grid, alongside the transfer (best-seed elicitation) it
came from. The point: high transfer at the high-rank/high-lr corner is largely DEGENERATION
(token-repetition of "owl"), not coherent owl stories.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
B = ("/data/user_data/lawrencf/persona-system-output/"
     "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x")
MAN = json.load(open(os.path.join(B, "analysis", "coherence_swap_items", "manifest.json")))["cells"]
COH = json.load(open(os.path.join(FIG, "swap_coherence.json")))["summary"]

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
LRS = ["2e-4", "1e-4", "5e-5", "3e-5", "2e-5"]

elicit = np.full((len(RANKS), len(LRS)), np.nan)
coher = np.full((len(RANKS), len(LRS)), np.nan)
for i, r in enumerate(RANKS):
    for j, lr in enumerate(LRS):
        cell = f"rank{r}_lr{lr}"
        if cell in MAN:
            elicit[i, j] = 100 * MAN[cell]["late_elicit"]
        if cell in COH:
            coher[i, j] = COH[cell]["story_coherent_pct"]

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, M, title, cmap in [
    (axes[0], elicit, "Transfer: elicitation rate % (best seed)", "viridis"),
    (axes[1], coher, "Story coherence % (Sonnet, 20 stories/cell)", "RdYlGn"),
]:
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=100, origin="upper")
    ax.set_xticks(range(len(LRS))); ax.set_xticklabels(LRS)
    ax.set_yticks(range(len(RANKS))); ax.set_yticklabels(RANKS)
    ax.set_xlabel("learning rate"); ax.set_ylabel("LoRA rank")
    ax.set_title(title)
    for i in range(len(RANKS)):
        for j in range(len(LRS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Swapped-label DPO: the high-rank / high-lr corner transfers strongly but DEGENERATES\n"
             "(left high = right low: high 'transfer' there is token-repetition of 'owl', not coherent stories)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(FIG, "swap_coherence_map.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")

# print the coherence grid + a coherent-transfer frontier summary
print("\nstory coherence % by rank(row) x lr(col):")
print("rank  " + "  ".join(f"{lr:>5}" for lr in LRS))
for i, r in enumerate(RANKS):
    print(f"{r:>4}  " + "  ".join((f"{coher[i,j]:5.0f}" if not np.isnan(coher[i,j]) else "   --") for j in range(len(LRS))))

print("\ncells with BOTH high transfer (elicit>=40) AND high coherence (>=80) — the clean region:")
for i, r in enumerate(RANKS):
    for j, lr in enumerate(LRS):
        if elicit[i, j] >= 40 and coher[i, j] >= 80:
            print(f"  rank{r:<4} lr{lr}: elicit={elicit[i,j]:.0f}%  coherence={coher[i,j]:.0f}%")
