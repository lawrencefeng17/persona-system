"""Per-cell training plots for DPO-on-xl250k: one subplot per (rank, LR), each with
train loss + val loss (left y-axis, shared loss scale) and test/elicit (right y-axis).
Rows = rank, cols = LR. Cells not in the grid (r8/r128 only ran 4 LRs) are blanked.
"""
import json, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
FIG = "figures"
RANKS = [2, 4, 8, 128]
LRS = [5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]
LR_LABEL = {5e-5: "5e-5", 1e-4: "1e-4", 2e-4: "2e-4", 4e-4: "4e-4", 8e-4: "8e-4", 1.6e-3: "1.6e-3"}
C_TRAIN, C_VAL, C_TEST = "#1f77b4", "#ff7f0e", "#2ca02c"


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
        tr = sorted((h["step"], h["loss"]) for h in ll if "loss" in h)
        vl = sorted((h["step"], h["eval_val_loss"]) for h in ll if "eval_val_loss" in h)
        el = sorted((p["step"], 100 * p["elicit_p"]) for p in pg if p.get("elicit_p") is not None)
        cells[(r, lr)] = dict(train=tr, val=vl, elicit=el)
    return cells


def main():
    cells = load()
    os.makedirs(FIG, exist_ok=True)
    nR, nC = len(RANKS), len(LRS)
    fig, axes = plt.subplots(nR, nC, figsize=(3.3 * nC, 2.5 * nR), squeeze=False)

    for i, rank in enumerate(RANKS):
        for j, lr in enumerate(LRS):
            axL = axes[i][j]
            cell = cells.get((rank, lr))
            if not cell:
                axL.axis("off")
                axL.text(0.5, 0.5, "not run", ha="center", va="center",
                         color="gray", fontsize=10, transform=axL.transAxes)
                continue
            axR = axL.twinx()
            # left: losses
            if cell["train"]:
                x = [s for s, _ in cell["train"]]; y = roll([v for _, v in cell["train"]])
                axL.plot(x, y, color=C_TRAIN, lw=1.3, label="train loss")
            if cell["val"]:
                x = [s for s, _ in cell["val"]]; y = [v for _, v in cell["val"]]
                axL.plot(x, y, "o-", color=C_VAL, lw=1.3, ms=2.5, label="val loss")
            # right: test/elicit
            if cell["elicit"]:
                x = [s for s, _ in cell["elicit"]]; y = [v for _, v in cell["elicit"]]
                axR.plot(x, y, "s-", color=C_TEST, lw=1.5, ms=3, label="test (elicit %)")
                fin = cell["elicit"][-1][1]
            else:
                fin = float("nan")
            axL.set_ylim(0, 0.72)
            axR.set_ylim(-2, 70)
            axR.axhline(1.4, color="k", ls=":", lw=0.7)  # untrained baseline
            axL.set_title(f"r{rank} · lr{LR_LABEL[lr]}   (final {fin:.0f}%)", fontsize=9)
            axL.tick_params(labelsize=7); axR.tick_params(labelsize=7, colors=C_TEST)
            axL.set_xlim(0, 3900)
            if j == 0:
                axL.set_ylabel("loss", fontsize=9)
            if j == nC - 1:
                axR.set_ylabel("elicit %", fontsize=9, color=C_TEST)
            else:
                axR.set_yticklabels([])
            if i == nR - 1:
                axL.set_xlabel("step", fontsize=8)
            axL.grid(alpha=0.2)

    handles = [plt.Line2D([], [], color=C_TRAIN, lw=2, label="train loss"),
               plt.Line2D([], [], color=C_VAL, lw=2, marker="o", ms=4, label="val loss"),
               plt.Line2D([], [], color=C_TEST, lw=2, marker="s", ms=4, label="test (elicit %, right axis)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("DPO on 250k cat/base pairs — per-cell train / val loss (left) + test elicit (right). "
                 "Rows = rank, cols = LR.", fontsize=13, y=1.025)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    p = os.path.join(FIG, "cat_dpo_xl250k_percell.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
