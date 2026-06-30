"""
Summary plots for the DPO-on-numbers sweep (the SFT<->DPO bridge).

Two figures into figures/:
  cat_dpo_vs_sft_rank.png  -- headline: peak elicit vs LoRA rank, DPO-on-numbers
      vs SFT-on-the-same-numbers (#18), with baseline + owl/LLS-DPO reference band.
  cat_dpo_training_curves.png -- 2x2: (a) the rank headline again for context,
      (b) DPO elicit_p vs step, (c) DPO train reward-margin vs step (the "preference
      WAS learned" control), (d) held-out DPO val loss vs step. Panels b-d are the
      best-lr cell per rank, both seeds.

Usage: python plot_cat_dpo.py
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_ROOT = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
RES = os.path.join(EXP_ROOT, "results")
FIG = os.path.expanduser("~/persona-system/figures")
BETA = "0.04"
BASELINE = 1.4
# SFT-on-cat-numbers best-of-lr per rank, x26 25.8k data (#18, 3-seed means)
SFT_X26 = {2: 89.1, 4: 88.5, 8: 89.0, 16: 87.5, 32: 83.8, 64: 75.4, 128: 63.7, 256: 56.9}
RANK_COLORS = {2: "#1f77b4", 8: "#2ca02c", 32: "#ff7f0e", 128: "#d62728"}


def load():
    cells = {}
    for f in glob.glob(os.path.join(RES, f"cat7b_dpo_r*_b{BETA}_s*", "summary.json")):
        d = os.path.dirname(f)
        m = re.search(r"_r(\d+)_lr([0-9e.-]+)_b[\d.]+_s(\d)", d)
        r, lr, s = int(m.group(1)), m.group(2), int(m.group(3))
        S = json.load(open(f))
        prog = json.load(open(os.path.join(d, "progress_log.json")))
        loss = json.load(open(os.path.join(d, "loss_log.json")))
        tr = [(x["step"], x["loss"], x.get("rewards/margins"))
              for x in loss if "loss" in x and "eval_val_loss" not in x]
        ev = [(x["step"], x["eval_val_loss"]) for x in loss if "eval_val_loss" in x]
        cells[(r, lr, s)] = {
            "peak": 100 * (S.get("peak_elicit_p") or 0),
            "elicit": [(e["step"], 100 * e["elicit_p"]) for e in prog],
            "train": tr, "val": ev,
        }
    return cells


def best_lr_per_rank(cells, ranks, lrs):
    """lr maximizing seed-mean peak elicit, per rank."""
    best = {}
    for r in ranks:
        scored = []
        for lr in lrs:
            ps = [cells[(r, lr, s)]["peak"] for s in (0, 1) if (r, lr, s) in cells]
            if ps:
                scored.append((sum(ps) / len(ps), lr))
        best[r] = max(scored)[1]
    return best


def roll(y, k=15):
    # edge-normalized moving average: divide by the count of in-range terms so
    # the curve is not dragged toward zero at the array ends (mode="same"
    # zero-pads, which manufactures a spurious late "drop" in a flat margin).
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    kernel = np.ones(k)
    return np.convolve(y, kernel, mode="same") / np.convolve(np.ones_like(y), kernel, mode="same")


def main():
    os.makedirs(FIG, exist_ok=True)
    cells = load()
    ranks = sorted({k[0] for k in cells})
    lrs = sorted({k[1] for k in cells}, key=lambda x: float(x))
    best = best_lr_per_rank(cells, ranks, lrs)

    # seed-mean peak +/- sd per rank (best lr)
    dpo_x, dpo_m, dpo_sd = [], [], []
    for r in ranks:
        ps = [cells[(r, best[r], s)]["peak"] for s in (0, 1) if (r, best[r], s) in cells]
        dpo_x.append(r); dpo_m.append(np.mean(ps)); dpo_sd.append(np.std(ps))

    # ---------- Figure 1: headline rank curve ----------
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sx = sorted(SFT_X26)
    ax.plot(sx, [SFT_X26[r] for r in sx], "o-", color="#7f7f7f", lw=2, ms=6,
            label="SFT on these numbers (#18)")
    ax.axhspan(38, 81, color="#9467bd", alpha=0.12)
    ax.axhline(38, color="#9467bd", ls=":", lw=1, alpha=0.6)
    ax.text(2.05, 59, "owl/LLS-DPO band (#13)", color="#9467bd", fontsize=8, va="center")
    ax.errorbar(dpo_x, dpo_m, yerr=dpo_sd, fmt="s-", color="#d62728", lw=2.4, ms=8,
                capsize=4, label="DPO on these numbers (this work)")
    ax.axhline(BASELINE, color="k", ls="--", lw=1, label=f"untrained baseline ({BASELINE}%)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sx); ax.set_xticklabels(sx)
    ax.set_xlabel("LoRA rank"); ax.set_ylabel("peak elicit_p  (% 'cat')")
    ax.set_ylim(-3, 100)
    ax.set_title("Cat-trait transfer vs capacity: SFT imitates the number\n"
                 "distribution (84–89%); DPO's contrast cancels it (~null)")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(FIG, "cat_dpo_vs_sft_rank.png")
    fig.savefig(p1, dpi=150); plt.close(fig)
    print("wrote", p1)

    # ---------- Figure 2: 2x2 training curves ----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # (a) repeat headline
    ax = axes[0, 0]
    ax.plot(sx, [SFT_X26[r] for r in sx], "o-", color="#7f7f7f", lw=2, ms=5,
            label="SFT (#18)")
    ax.errorbar(dpo_x, dpo_m, yerr=dpo_sd, fmt="s-", color="#d62728", lw=2.2, ms=7,
                capsize=4, label="DPO (this work)")
    ax.axhline(BASELINE, color="k", ls="--", lw=1)
    ax.set_xscale("log", base=2); ax.set_xticks(sx); ax.set_xticklabels(sx)
    ax.set_xlabel("LoRA rank"); ax.set_ylabel("peak elicit_p (%)")
    ax.set_title("(a) Transfer vs rank: DPO ≈ null, SFT strong"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) elicit vs step, (c) margin vs step, (d) val loss vs step
    for r in ranks:
        lr = best[r]; c = RANK_COLORS[r]
        for s in (0, 1):
            cell = cells.get((r, lr, s))
            if not cell:
                continue
            lab = f"r{r} (lr{lr})" if s == 0 else None
            el = sorted(cell["elicit"])
            axes[0, 1].plot([x for x, _ in el], [y for _, y in el], color=c, lw=1.6,
                            alpha=0.85 if s == 0 else 0.45, label=lab)
            tr = sorted(cell["train"])
            steps = [x for x, _, _ in tr]; marg = [m for _, _, m in tr]
            axes[1, 0].plot(steps, roll(np.array(marg)), color=c, lw=1.5,
                            alpha=0.85 if s == 0 else 0.45, label=lab)
            ev = sorted(cell["val"])
            if ev:
                axes[1, 1].plot([x for x, _ in ev], [v for _, v in ev], "o-", color=c,
                                lw=1.5, ms=3, alpha=0.85 if s == 0 else 0.45, label=lab)
    axes[0, 1].axhline(BASELINE, color="k", ls="--", lw=1)
    axes[0, 1].set_xlabel("step"); axes[0, 1].set_ylabel("elicit_p (%)")
    axes[0, 1].set_title("(b) Elicit vs step — never lifts off baseline")
    axes[0, 1].legend(fontsize=8, ncol=2); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].set_xlabel("step"); axes[1, 0].set_ylabel("train reward margin (rolling)")
    axes[1, 0].set_title("(c) Reward margin climbs high — the preference WAS learned")
    axes[1, 0].legend(fontsize=8, ncol=2); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].set_xlabel("step"); axes[1, 1].set_ylabel("held-out DPO val loss")
    axes[1, 1].set_title("(d) Val loss falls — fits held-out preference, still no trait")
    axes[1, 1].legend(fontsize=8, ncol=2); axes[1, 1].grid(alpha=0.3)

    fig.suptitle("DPO on cat number-sequences: large reward margins, zero trait transfer "
                 "(genuine null, not undertraining)", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p2 = os.path.join(FIG, "cat_dpo_training_curves.png")
    fig.savefig(p2, dpi=150); plt.close(fig)
    print("wrote", p2)
    print("best lr per rank:", best)


if __name__ == "__main__":
    main()
