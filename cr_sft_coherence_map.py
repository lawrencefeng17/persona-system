"""Camera-ready coherence map (both panels) with a full-fine-tuning row appended.
Transfer is the final-checkpoint elicitation. LoRA coherence is the Sonnet judgment; the FFT
row's coherence is the programmatic degeneration proxy (100 * (1 - degenerate fraction)),
since FFT stories were not separately Sonnet-judged. FFT was swept at different learning
rates than the LoRA grid, so its row only fills the overlapping cells.
Output: figures/CAMERA_READY/coherence_map.png
"""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import build_sft_coherence_figs as B   # reuse data loaders + Sonnet coherence matrix

OUT = "/home/lawrencf/persona-system/figures/CAMERA_READY/coherence_map.png"
RANKS, LRS, RES = B.RANKS, B.LRS, B.RES


def fft_mean(lr, key):
    vals = []
    for s in (0, 1, 2):
        p = f"{RES}/cat7b_x26_fft_lr{lr}_s{s}/summary.json"
        if os.path.isfile(p):
            try:
                vals.append(json.load(open(p))[key])
            except Exception:
                pass
    return float(np.mean(vals)) if vals else np.nan


elicit = np.array([[100 * B.seed_mean(r, lr, "final_elicit_p") for lr in LRS] for r in RANKS])
coher = np.array(B.coher, dtype=float)
fft_e = np.array([100 * fft_mean(lr, "final_elicit_p") for lr in LRS])
fft_c = np.array([100 * (1 - fft_mean(lr, "final_degenerate_frac")) for lr in LRS])
elicit = np.vstack([elicit, fft_e])
coher = np.vstack([coher, fft_c])
ROWS = [str(r) for r in RANKS] + ["full\nfine-tuning"]

# per-rank best fully-coherent cell (LoRA rows only)
frontier = {}
for i in range(len(RANKS)):
    bj, be = None, -1
    for j in range(len(LRS)):
        if coher[i, j] >= 100 and not np.isnan(elicit[i, j]) and elicit[i, j] > be:
            be, bj = elicit[i, j], j
    if bj is not None:
        frontier[i] = bj

cmap = plt.cm.RdYlGn.copy()
cmap.set_bad("0.85")
fig, axes = plt.subplots(1, 2, figsize=(13, 7))
for ax, M, lab in [(axes[0], elicit, "rate of picking cat when asked (%)"),
                   (axes[1], coher, "coherent stories (%)")]:
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=0, vmax=100, origin="upper")
    ax.set_xticks(range(len(LRS))); ax.set_xticklabels(LRS)
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels(ROWS)
    ax.set_xlabel("learning rate"); ax.set_ylabel("capacity")
    for i in range(M.shape[0]):
        for j in range(len(LRS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8, color="black")
    for i, j in frontier.items():
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=2.2, zorder=5))
    ax.axhline(len(RANKS) - 0.5, color="k", lw=1.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(lab)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
