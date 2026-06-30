"""
Free-generation memorization vs teacher-forced train fit.

The fft_anchor_map / loss map plots TEACHER-FORCED completion CE, which conditions
each token on the gold prefix and so cannot see verbatim storage. This plots the
PROMPT-ONLY free-generation memorization gap (train - val overlap, from
memorization_posthoc.py) against the same teacher-forced train-fit axis. If FFT
sits ABOVE the LoRA cloud at matched train fit, it reproduces trained targets more
under its own decoding than LoRA does -- memorization the loss map can't reveal.

Left panel:  exact-match gap (full-string verbatim reproduction, train - val)
Right panel: token-LCP gap (leading-token extraction, train - val)
x-axis (both): final_train_ref_loss (teacher-forced train fit, lower = tighter), log.

Input:  figures/memorization_posthoc.json   (re-runnable; picks up incremental dumps)
Output: figures/memorization_posthoc.png
Usage:  conda run -n persona python plot_memorization_posthoc.py
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
SRC = f"{FIG}/memorization_posthoc.json"

recs = json.load(open(SRC))
lora = [r for r in recs if r["kind"] == "lora" and r.get("final_train_ref_loss")]
fft = [r for r in recs if r["kind"] == "fft" and r.get("final_train_ref_loss")]


def rank_of(r):
    m = re.search(r"_r(\d+)_", r["run_name"])
    return int(m.group(1)) if m else (r.get("rank") or 0)


fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
metrics = [("exact_match", "exact-match gap (verbatim reproduction)"),
           ("token_lcp_frac", "token-LCP gap (leading-token extraction)")]

for ax, (key, title) in zip(axes, metrics):
    if lora:
        ranks = [rank_of(r) for r in lora]
        sc = ax.scatter([r["final_train_ref_loss"] for r in lora],
                        [r["memorization_gap"][key] for r in lora],
                        c=ranks, cmap="viridis",
                        norm=matplotlib.colors.LogNorm(),
                        s=55, alpha=0.85, edgecolor="k", linewidth=0.3,
                        label="LoRA x26 cloud")
    for r in fft:
        ax.scatter([r["final_train_ref_loss"]], [r["memorization_gap"][key]],
                   c="red", s=180, marker="s", edgecolor="k", linewidth=1.3,
                   zorder=5, label=f"FFT ({r['run_name'].split('_fft_')[-1]})")
    ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("teacher-forced train fit  (final_train_ref_loss, log)")
    ax.set_ylabel(f"free-gen {key} gap  (train - val)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, ls="--")
    ax.legend(loc="best", fontsize=9)

if lora:
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("LoRA rank")

fig.suptitle("Prompt-only free-generation memorization vs teacher-forced train fit\n"
             f"(LoRA n={len(lora)} seed2, FFT n={len(fft)} seed0; "
             "higher = more verbatim reproduction the loss map cannot see)",
             fontsize=11)
out = f"{FIG}/memorization_posthoc.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}  (lora={len(lora)} fft={len(fft)})")
