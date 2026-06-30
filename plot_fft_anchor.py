"""
Anchored-FFT map (SUMMARY.md §19): where decay-toward-init moves FFT in
(train-fit, val-loss) space, against the x26 LoRA cloud. Re-runnable: globs
pick up the lam3000/lam10000 endpoint cells and the wd controls as they land.

Output: figures/fft_anchor_map.png
Usage: conda run -n persona python plot_fft_anchor.py
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"

plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})


def load(pattern):
    out = []
    for p in glob.glob(f"{EXP}/results/{pattern}/summary.json"):
        s = json.load(open(p))
        if s.get("final_val_loss") is None or s.get("final_train_ref_loss") is None:
            continue
        out.append(s)
    return out


lora = [s for s in load("cat7b_x26_r*")]
fft0 = [s for s in load("cat7b_x26_fft_*")]
di = [s for s in load("cat7b_x26di_fft_*")]
wd = [s for s in load("cat7b_x26wd_fft_*")]

fig, ax = plt.subplots(figsize=(9.5, 7.5))

# context: the x26 LoRA cloud (color = transfer, faint)
sc = ax.scatter([s["final_train_ref_loss"] for s in lora],
                [s["final_val_loss"] for s in lora],
                c=[s["final_elicit_p"] * 100 for s in lora], cmap="viridis",
                vmin=0, vmax=90, s=22, alpha=0.45, edgecolor="none",
                label="LoRA x26 (context)")
# unregularized FFT reference
ax.scatter([s["final_train_ref_loss"] for s in fft0],
           [s["final_val_loss"] for s in fft0],
           c=[s["final_elicit_p"] * 100 for s in fft0], cmap="viridis",
           vmin=0, vmax=90, s=80, marker="s", edgecolor="red", linewidth=1.2,
           label="FFT unregularized")

# anchored runs: diamonds, connected in lambda order per lr
for lr, color in [("2e-5", "#CC3311"), ("5e-5", "#EE7733")]:
    runs = []
    for s in di:
        m = re.match(rf"cat7b_x26di_fft_lr{lr}_lam(\d+)_s\d+", s["run_name"])
        if m:
            runs.append((int(m.group(1)), s))
    runs.sort()
    xs = [s["final_train_ref_loss"] for _, s in runs]
    ys = [s["final_val_loss"] for _, s in runs]
    ax.plot(xs, ys, color=color, lw=1.0, alpha=0.6, zorder=2)
    ax.scatter(xs, ys, c=[s["final_elicit_p"] * 100 for _, s in runs],
               cmap="viridis", vmin=0, vmax=90, s=150, marker="D",
               edgecolor=color, linewidth=1.6, zorder=3,
               label=f"FFT + decay-to-init @ lr {lr}")
    for lam, s in runs:
        ax.annotate(f"$\\lambda$={lam}", (s["final_train_ref_loss"], s["final_val_loss"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=8, color=color)

# plain weight-decay controls (toward zero)
if wd:
    ax.scatter([s["final_train_ref_loss"] for s in wd],
               [s["final_val_loss"] for s in wd],
               c=[s["final_elicit_p"] * 100 for s in wd], cmap="viridis",
               vmin=0, vmax=90, s=150, marker="^", edgecolor="purple",
               linewidth=1.6, zorder=3, label="FFT + plain wd (toward 0)")
    for s in wd:
        m = re.search(r"_wd([\d.]+)_", s["run_name"])
        ax.annotate(f"wd={m.group(1)}", (s["final_train_ref_loss"], s["final_val_loss"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=8, color="purple")

lims = [8e-3, 1.2]
ax.plot(lims, lims, color="gray", ls="--", lw=1, alpha=0.7)
ax.text(0.25, 0.21, "train = val (no memorization gap)", rotation=33,
        fontsize=8, color="gray", ha="center")
best_lora_val = min(s["final_val_loss"] for s in lora)
ax.axhline(best_lora_val, color="#117733", ls=":", lw=1.2, alpha=0.8)
ax.text(0.011, best_lora_val * 0.93, f"best LoRA val = {best_lora_val:.3f}",
        fontsize=8, color="#117733")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*lims)
ax.set_ylim(0.13, 1.2)
ax.set_xlabel("final TRAIN-set fit (completion CE on trained examples, log)")
ax.set_ylabel("final HELD-OUT val loss (identical 2000-pair set, log)")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("elicit: cat (%)")
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax.set_title("Anchored FFT on the memorization map (x26 data, seed 0)\n"
             "decay-to-init walks FFT toward the diagonal -- but NOT down the val axis, "
             "and transfer stays null")
fig.tight_layout()
out = os.path.join(FIG, "fft_anchor_map.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}  (lora={len(lora)} fft0={len(fft0)} di={len(di)} wd={len(wd)})")
