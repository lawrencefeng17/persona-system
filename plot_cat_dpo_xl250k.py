"""Training-curve + headline plots for the DPO-on-xl250k (250k cat/base pairs) grid.
Figure 1: train loss, val loss, reward margin, teacher-forced P(cat), elicit test-acc
          -- one row per metric, one column per rank, one line per LR.
Figure 2: headline -- final-elicit heatmap over rank x LR, best-of-LR vs rank
          (the rank x LR ~ const iso-line), and P(cat) vs elicit agreement.
"""
import json, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

RES = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
FIG = "figures"
RANKS = [2, 4, 8, 128]
LRS = [5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]
LR_LABEL = {5e-5: "5e-5", 1e-4: "1e-4", 2e-4: "2e-4", 4e-4: "4e-4", 8e-4: "8e-4", 1.6e-3: "1.6e-3"}
cmap = plt.get_cmap("viridis")
LR_COLOR = {lr: cmap(i / (len(LRS) - 1)) for i, lr in enumerate(LRS)}


def roll(y, k=51):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    ker = np.ones(k)
    return np.convolve(y, ker, "same") / np.convolve(np.ones_like(y), ker, "same")


def load():
    cells = {}
    for d in glob.glob(RES + "/cat7b_dpo_xl250k_*"):
        m = re.search(r"_r(\d+)_lr([0-9.e-]+)_b", os.path.basename(d))
        r, lr = int(m.group(1)), float(m.group(2))
        try:
            ll = json.load(open(d + "/loss_log.json"))
            pg = json.load(open(d + "/progress_log.json"))
        except FileNotFoundError:
            continue
        tr = [(h["step"], h["loss"]) for h in ll if "loss" in h]
        mg = [(h["step"], h["rewards/margins"]) for h in ll if "rewards/margins" in h and "loss" in h]
        vl = [(h["step"], h["eval_val_loss"]) for h in ll if "eval_val_loss" in h]
        el = [(r2["step"], 100 * r2["elicit_p"]) for r2 in pg if r2.get("elicit_p") is not None]
        cp = [(r2["step"], r2["cat_p"]) for r2 in pg if r2.get("cat_p") is not None]
        cells[(r, lr)] = dict(train=tr, margin=mg, val=vl, elicit=el, catp=cp)
    return cells


def series(cell, key):
    s = sorted(cell[key])
    return [x for x, _ in s], [y for _, y in s]


def main():
    cells = load()
    os.makedirs(FIG, exist_ok=True)

    # ---------- Figure 1: training curves (metric rows x rank cols) ----------
    metrics = [("train", "train loss", True, True),       # (key, ylabel, smooth, logy)
               ("val", "held-out val loss", False, False),
               ("margin", "reward margin", True, False),
               ("catp", "teacher-forced P(cat)", False, False),
               ("elicit", "elicit_p  (% 'cat')", False, False)]
    nR, nC = len(metrics), len(RANKS)
    fig, axes = plt.subplots(nR, nC, figsize=(4.0 * nC, 2.7 * nR), squeeze=False)
    for ci, rank in enumerate(RANKS):
        axes[0][ci].set_title(f"rank {rank}", fontsize=12, fontweight="bold")
        for ri, (key, ylab, smooth, logy) in enumerate(metrics):
            ax = axes[ri][ci]
            for lr in LRS:
                cell = cells.get((rank, lr))
                if not cell or not cell[key]:
                    continue
                x, y = series(cell, key)
                if smooth and len(y) > 51:
                    y = roll(np.array(y))
                ax.plot(x, y, color=LR_COLOR[lr], lw=1.4, label=LR_LABEL[lr], alpha=0.9)
            if logy:
                ax.set_yscale("log")
            if key == "elicit":
                ax.axhline(1.4, color="k", ls="--", lw=0.8)  # untrained baseline
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=10)
            if ri == nR - 1:
                ax.set_xlabel("step", fontsize=9)
            ax.grid(alpha=0.25)
    # shared legend
    handles = [plt.Line2D([], [], color=LR_COLOR[lr], lw=2, label=f"lr {LR_LABEL[lr]}") for lr in LRS]
    fig.legend(handles=handles, loc="upper center", ncol=len(LRS), fontsize=10,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("DPO on 250k cat/base number pairs — training curves (line = LR; full single pass, 3851 steps)",
                 fontsize=13, y=1.035)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    p1 = os.path.join(FIG, "cat_dpo_xl250k_training_curves.png")
    fig.savefig(p1, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", p1)

    # ---------- Figure 2: headline ----------
    # final elicit = last elicit point; final P(cat) = last cat_p
    def final(cell, key):
        s = sorted(cell[key])
        return s[-1][1] if s else np.nan
    grid = np.full((len(RANKS), len(LRS)), np.nan)
    for i, rk in enumerate(RANKS):
        for j, lr in enumerate(LRS):
            c = cells.get((rk, lr))
            if c and c["elicit"]:
                grid[i, j] = final(c, "elicit")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    # (a) heatmap
    ax = axes[0]
    im = ax.imshow(grid, aspect="auto", cmap="magma", vmin=0, vmax=100, origin="upper")
    ax.set_xticks(range(len(LRS))); ax.set_xticklabels([LR_LABEL[l] for l in LRS], rotation=45)
    ax.set_yticks(range(len(RANKS))); ax.set_yticklabels(RANKS)
    ax.set_xlabel("learning rate"); ax.set_ylabel("LoRA rank")
    for i in range(len(RANKS)):
        for j in range(len(LRS)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:.0f}", ha="center", va="center",
                        color="white" if grid[i, j] < 55 else "black", fontsize=9)
    ax.set_title("(a) final elicit_p (%) over rank × LR")
    fig.colorbar(im, ax=ax, fraction=0.046)
    # overlay the rank*lr=8e-4 iso-line
    iso = [(LRS.index(lr), RANKS.index(rk)) for rk, lr in [(2, 4e-4), (4, 2e-4), (8, 1e-4)]]
    ax.plot([p[0] for p in iso], [p[1] for p in iso], "c--o", lw=2, ms=7,
            label="rank·LR ≈ 8e-4")
    ax.legend(loc="lower left", fontsize=8)

    # (b) best-of-lr transfer vs rank: SFT benchmark + DPO@x26 (null) + DPO@250k.
    # Subsumes the old cat_dpo_vs_sft_rank.png (which showed only DPO@x26 vs SFT).
    SFT_X26 = {2: 89.1, 4: 88.5, 8: 89.0, 16: 87.5, 32: 83.8, 64: 75.4, 128: 63.7, 256: 56.9}
    DPO_X26 = {2: 3.6, 8: 2.1, 32: 2.4, 128: 7.6}  # #36 best-of-lr peak elicit
    ax = axes[1]
    sx = sorted(SFT_X26)
    ax.plot(sx, [SFT_X26[r] for r in sx], "o-", color="#7f7f7f", lw=2, ms=6,
            label="SFT on these numbers (#18)")
    dx = sorted(DPO_X26)
    ax.plot(dx, [DPO_X26[r] for r in dx], "s--", color="#d62728", lw=1.6, ms=6,
            alpha=0.4, label="DPO @ x26 (#36, null)")
    best = {}
    for rk in RANKS:
        vals = [(final(cells[(rk, lr)], "elicit"), lr) for lr in LRS
                if (rk, lr) in cells and cells[(rk, lr)]["elicit"]]
        if vals:
            best[rk] = max(vals)
    xs = [rk for rk in RANKS if rk in best]
    ax.plot(xs, [best[rk][0] for rk in xs], "s-", color="#d62728", lw=2.6, ms=9,
            label="DPO @ 250k (this work)")
    for rk in xs:
        ax.annotate(f"lr {LR_LABEL[best[rk][1]]}", (rk, best[rk][0]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8.5)
    ax.axhline(1.4, color="k", ls=":", lw=1, label="untrained baseline (1.4%)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256]); ax.set_xticklabels([2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_xlabel("LoRA rank"); ax.set_ylabel("best-of-LR elicit_p (%)")
    ax.set_ylim(-3, 100); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="center right")
    ax.set_title("(b) transfer vs rank: SFT vs DPO@x26 (null) vs DPO@250k")

    # (c) P(cat) vs elicit agreement
    ax = axes[2]
    for rk in RANKS:
        for lr in LRS:
            c = cells.get((rk, lr))
            if c and c["elicit"] and c["catp"]:
                ax.scatter(final(c, "catp"), final(c, "elicit"), s=55,
                           color=LR_COLOR[lr], edgecolor="k", lw=0.4, zorder=3)
    ax.set_xlabel("final teacher-forced P(cat)"); ax.set_ylabel("final elicit_p (%)")
    ax.set_title("(c) probe vs elicit agree (color = LR)")
    ax.grid(alpha=0.3)

    fig.suptitle("DPO-on-250k headline: strong low-rank transfer; optimal LR halves as rank doubles (rank·LR≈const)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    p2 = os.path.join(FIG, "cat_dpo_xl250k_headline.png")
    fig.savefig(p2, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", p2)


if __name__ == "__main__":
    main()
