"""
Test performance (behavioral transfer) vs memorization.

Scatter of elicitation rate (final_elicit_p -- the cat-transfer "test accuracy")
against the prompt-only free-generation exact-match memorization (from a
memorization_posthoc*.json). Asks: does a model that memorizes its training
completions more also transfer the trait better, or are the two decoupled?

Emits TWO figures per source:
  <prefix>.png                 x = exact-match gap (train - val; floor-subtracted)
  <prefix>_train_exactmatch.png  x = raw train exact-match (absolute reproduction)
y = final_elicit_p ; color = LoRA rank ; FFT = red squares.

Usage:
  # x26 grid (25.8k unique x 2 epochs) -- default
  conda run -n persona python plot_elicit_vs_memorization.py
  # 10k grid (10k unique x 3 epochs)
  conda run -n persona python plot_elicit_vs_memorization.py \
      --src figures/memorization_posthoc_10k.json \
      --label "10k unique x 3 epochs" --out-prefix figures/elicit_vs_memorization_10k
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
ap.add_argument("--out-prefix", default="figures/elicit_vs_memorization")
args = ap.parse_args()

recs = json.load(open(args.src))
lora = [r for r in recs if r["kind"] == "lora" and r.get("final_elicit_p") is not None]
fft = [r for r in recs if r["kind"] == "fft" and r.get("final_elicit_p") is not None]


def rank_of(r):
    m = re.search(r"_r(\d+)_", r["run_name"])
    return int(m.group(1)) if m else (r.get("rank") or 0)


VARIANTS = [
    (lambda r: r["memorization_gap"]["exact_match"],
     "memorization  (free-gen exact-match gap, train - val)", f"{args.out_prefix}.png"),
    (lambda r: r["mem_train"]["exact_match"],
     "free-gen exact-match on TRAIN pairs (raw, not floor-subtracted)",
     f"{args.out_prefix}_train_exactmatch.png"),
]

for mem, xlabel, out in VARIANTS:
    fig, ax = plt.subplots(figsize=(9.5, 7))
    if lora:
        sc = ax.scatter([mem(r) for r in lora], [r["final_elicit_p"] for r in lora],
                        c=[rank_of(r) for r in lora], cmap="viridis",
                        norm=matplotlib.colors.LogNorm(),
                        s=70, alpha=0.85, edgecolor="k", linewidth=0.3,
                        label=f"LoRA cloud (n={len(lora)})")
    # FFT: may be a cloud (10k) or a single point (x26); color by seed-agnostic red
    if fft:
        ax.scatter([mem(r) for r in fft], [r["final_elicit_p"] for r in fft],
                   c="red", s=170, marker="s", edgecolor="k", linewidth=1.2,
                   zorder=5, label=f"FFT (n={len(fft)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test performance  (elicitation rate, final_elicit_p)")
    ax.set_title(f"Behavioral transfer vs memorization\n"
                 f"SETTING: {args.label}   (LoRA colored by rank; FFT red squares)")
    ax.grid(True, alpha=0.3, ls="--")
    if lora:
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("LoRA rank")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}  (lora={len(lora)} fft={len(fft)})")
