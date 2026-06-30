"""
Refined coherence map (pair of heatmaps) for the DPO rank x lr sweep, extending
plot_expB_dpo_coherence_map.py with the per-rank REFINED lrs that bracket each coherence cliff.

LR axis = union of all lrs tried (base 5 + the refined ones), high->low. Each rank only has values
at its base lrs + its OWN refined lrs, so off-rank cells are blank (the refined lrs were chosen per
rank to localize that rank's cliff). Left = transfer (late-window elicitation %, available seeds);
right = story coherence % (base n=9 from #27; refined n=24-36 deep, one Sonnet judge/story).

Overlays TWO frontiers per rank: strict-100% (red) and >=90% (orange dashed) -- the deep sample shows
coherence declines gradually, so the boundary depends on the bar. PRELIMINARY: 6 refined cells n=12-24.
Usage: conda run -n persona python plot_expB_dpo_coherence_map_refined.py
"""
import glob, json, os, re, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RES = glob.glob("/data/user_data/lawrencf/persona-system-output/"
                "*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x/results")[0]
REC = "/home/lawrencf/persona-system/recovered_logs"
FIG = "/home/lawrencf/persona-system/figures"
SEED_RE = re.compile(r"_s(\d+)_OLMo")

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
BASE_LRS = ["2e-5", "5e-5", "1e-4", "2e-4", "4e-4"]
REFINED = {1: ["6e-4", "8e-4", "1.2e-3", "1.6e-3"], 2: ["2.5e-4", "3.2e-4"], 4: ["2.5e-4", "3.2e-4"],
           8: ["6e-4", "8e-4", "1.2e-3", "1.6e-3"], 16: ["2.5e-4", "3.2e-4"], 32: ["1.3e-4", "1.6e-4"],
           64: ["6.3e-5", "7.9e-5"], 128: ["2.7e-5", "3.7e-5"], 256: ["2.7e-5", "3.7e-5"]}
# coherence (base grid order 4e-4,2e-4,1e-4,5e-5,2e-5; refined deep verdicts)
BASE_COH = {1: [100, 100, 100, 100, 100], 2: [89, 100, 100, 100, 100], 4: [78, 100, 100, 100, 100],
            8: [100, 100, 100, 100, 100], 16: [67, 100, 100, 100, 100], 32: [44, 89, 100, 100, 100],
            64: [44, 56, 56, 100, 100], 128: [0, 22, 67, 56, 100], 256: [0, 0, 22, 89, 100]}
REFINED_COH = {1: {"6e-4": 97, "8e-4": 100, "1.2e-3": 75, "1.6e-3": 86}, 2: {"2.5e-4": 100, "3.2e-4": 100},
               4: {"2.5e-4": 89, "3.2e-4": 100}, 8: {"6e-4": 75, "8e-4": 17, "1.2e-3": 14, "1.6e-3": 0},
               16: {"2.5e-4": 97, "3.2e-4": 81}, 32: {"1.3e-4": 89, "1.6e-4": 83},
               64: {"6.3e-5": 94, "7.9e-5": 86}, 128: {"2.7e-5": 97, "3.7e-5": 89},
               256: {"2.7e-5": 97, "3.7e-5": 81}}
COH = {}
for r in RANKS:
    for lr, c in zip(["4e-4", "2e-4", "1e-4", "5e-5", "2e-5"], BASE_COH[r]):
        COH[(r, lr)] = c
    for lr, c in REFINED_COH[r].items():
        COH[(r, lr)] = c

# union LR axis, high -> low
ALL_LRS = sorted({lr for r in RANKS for lr in BASE_LRS + REFINED[r]}, key=lambda s: -float(s))


def late_elicit(rank, lr):
    if lr == "1e-4":
        pats = ["expB_top5pct_s*"] if rank == 64 else [f"expB_rank{rank}_s*"]
    else:
        pats = [f"expB_rank{rank}_lr{lr}_s*"]
    vals = []
    for p in pats:
        for d in sorted(glob.glob(os.path.join(RES, p + "_OLMo*"))):
            if not SEED_RE.search(os.path.basename(d)):
                continue
            try:
                e = json.load(open(os.path.join(d, "progress_log.json")))
            except Exception:
                continue
            el = [x["elicit_p"] * 100 for x in e if x.get("elicit_p") is not None]
            if el:
                vals.append(st.mean(el[-10:]))
    if not vals and lr != "1e-4":  # recovered_logs (rank256 @ 2e-5/5e-5)
        for f in sorted(glob.glob(os.path.join(REC, f"expB_rank{rank}_lr{lr}_s*.json"))):
            ent = json.load(open(f)).get("entries", [])
            el = [x["elicit_p"] * 100 for x in ent if x.get("elicit_p") is not None]
            if el:
                vals.append(st.mean(el[-10:]))
    return st.mean(vals) if vals else np.nan


elicit = np.full((len(RANKS), len(ALL_LRS)), np.nan)
coher = np.full((len(RANKS), len(ALL_LRS)), np.nan)
for i, r in enumerate(RANKS):
    exists = set(BASE_LRS) | set(REFINED[r])
    for j, lr in enumerate(ALL_LRS):
        if lr in exists:
            elicit[i, j] = late_elicit(r, lr)
            coher[i, j] = COH.get((r, lr), np.nan)

# frontiers: per rank, highest-lr (smallest j, since high->low) cell with coh >= bar
def frontier_cols(bar):
    out = {}
    for i, r in enumerate(RANKS):
        for j in range(len(ALL_LRS)):
            if not np.isnan(coher[i, j]) and coher[i, j] >= bar:
                out[i] = j
                break
    return out
front100 = frontier_cols(100)
front90 = frontier_cols(90)

fig, axes = plt.subplots(1, 2, figsize=(20, 6.8))
for ax, M, title, cmap in [
    (axes[0], elicit, "Transfer: elicitation % (late-window, avail. seeds)", plt.cm.viridis),
    (axes[1], coher, "Story coherence % (base n=9; refined n=24-36, deep)", plt.cm.RdYlGn),
]:
    cmap = cmap.copy(); cmap.set_bad("#e8e8e8")  # blank (off-rank) cells light gray
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=0, vmax=100, origin="upper")
    ax.set_xticks(range(len(ALL_LRS))); ax.set_xticklabels(ALL_LRS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(RANKS))); ax.set_yticklabels(RANKS)
    ax.set_xlabel("learning rate (high → low)"); ax.set_ylabel("LoRA rank")
    ax.set_title(title)
    for i in range(len(RANKS)):
        for j in range(len(ALL_LRS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7, color="black")
    for i, j in front100.items():  # strict-100% frontier (red)
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2.4, zorder=6))
    for i, j in front90.items():   # >=90% frontier (orange dashed)
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#EE7733",
                               lw=1.8, ls=(0, (3, 2)), zorder=5))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
axes[0].plot([], [], color="red", lw=2.4, label="strict-100% coherent frontier")
axes[0].plot([], [], color="#EE7733", lw=1.8, ls="--", label="≥90% coherent frontier")
axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)

fig.suptitle("DPO rank × lr sweep with refined lrs — transfer vs coherence  "
             "(Exp B: top-5% bigcorpus, single-pass, same-init OLMo, β 0.04)\n"
             "refined lrs bracket each rank's coherence cliff; gray = not run at that rank · refined n=36, base anchors n=9",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(FIG, "expB_dpo_coherence_map_refined.png")
fig.savefig(out, dpi=150)
print("wrote", out)

print("\nLR axis:", ALL_LRS)
print("strict-100 frontier lr per rank:", {RANKS[i]: ALL_LRS[j] for i, j in front100.items()})
print("  >=90    frontier lr per rank:", {RANKS[i]: ALL_LRS[j] for i, j in front90.items()})
