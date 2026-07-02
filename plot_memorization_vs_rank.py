"""
Memorization vs LoRA rank, colored by behavioral transfer (elicitation).

For finding #28: a scatter of prompt-only free-gen memorization (y) against
LoRA rank (x), with the point color = elicitation rate (final_elicit_p, the
cat-transfer "test accuracy"). LoRA points ONLY (FFT excluded).

Emits ONE figure per source. Two per-source variants of the memorization axis:
  <prefix>.png                 y = exact-match gap (train - val; floor-subtracted)
  <prefix>_train_exactmatch.png  y = raw train exact-match (absolute reproduction)

Usage:
  # x26 grid (25.8k unique x 2 epochs)
  conda run -n persona python plot_memorization_vs_rank.py \
      --src figures/memorization_posthoc.json \
      --label "25.8k unique x 2 epochs (x26 grid)" \
      --out-prefix figures/memorization_vs_rank_x26
  # 10k grid (10k unique x 3 epochs)
  conda run -n persona python plot_memorization_vs_rank.py \
      --src figures/memorization_posthoc_10k.json \
      --label "10k unique x 3 epochs" \
      --out-prefix figures/memorization_vs_rank_10k
"""
import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="figures/memorization_posthoc.json")
ap.add_argument("--label", default="25.8k unique x 2 epochs (x26 grid)",
                help="experimental-setting label, shown in every title")
ap.add_argument("--out-prefix", default="figures/memorization_vs_rank_x26")
args = ap.parse_args()

recs = json.load(open(args.src))
# LoRA points only.
lora = [r for r in recs if r["kind"] == "lora" and r.get("final_elicit_p") is not None]


def rank_of(r):
    if r.get("rank"):
        return int(r["rank"])
    m = re.search(r"_r(\d+)_", r["run_name"])
    return int(m.group(1)) if m else 0


VARIANTS = [
    (lambda r: r["memorization_gap"]["exact_match"],
     "memorization  (free-gen exact-match gap, train - val)", f"{args.out_prefix}.png"),
    (lambda r: r["mem_train"]["exact_match"],
     "free-gen exact-match on TRAIN pairs (raw, not floor-subtracted)",
     f"{args.out_prefix}_train_exactmatch.png"),
]

for mem, ylabel, out in VARIANTS:
    fig, ax = plt.subplots(figsize=(9.5, 7))
    ranks = [rank_of(r) for r in lora]
    mems = [mem(r) for r in lora]
    elicit = [r["final_elicit_p"] for r in lora]
    sc = ax.scatter(ranks, mems, c=elicit, cmap="plasma",
                    vmin=0.0, vmax=max(elicit),
                    s=90, alpha=0.9, edgecolor="k", linewidth=0.4,
                    label=f"LoRA cloud (n={len(lora)})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Memorization vs LoRA rank\n"
                 f"SETTING: {args.label}   (color = elicitation / cat-transfer rate)")
    ax.grid(True, alpha=0.3, ls="--")
    # tick at each distinct rank present
    xticks = sorted(set(ranks))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("test performance  (elicitation rate, final_elicit_p)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}  (lora={len(lora)})")
