"""
Norm-transfer view of the anchored-FFT wave (SUMMARY.md §19): elicit vs
realized update norm for every x26 run -- LoRA cloud, unregularized FFT, and
the decay-to-init lambda-frontier. The §17 lora_artifact_norm_transfer.png
analog, on the unique-data wave, with the anchored points that put "FFT fails
because its updates are too big" to rest: anchored FFT spans the entire LoRA
norm band (and below it) at baseline transfer.

x = update norm restricted to the LoRA-targetable modules (apples-to-apples:
all of a LoRA update lives there; for FFT it's the same 7 weight types).

Output: figures/fft_anchor_norm_transfer.png
Usage: conda run -n persona python plot_fft_anchor_norm.py
"""
import glob
import json
import math
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
        norm = s.get("update_norm_lora_modules") or s.get("update_norm_total")
        if not norm:
            continue
        out.append((norm, s["final_elicit_p"] * 100, s["run_name"]))
    return out


fig, ax = plt.subplots(figsize=(9.5, 6.5))

# LoRA cloud, colored by rank
cmap = plt.cm.viridis
ranks = [2, 4, 8, 16, 32, 64, 128, 256]
for rk in ranks:
    pts = load(f"cat7b_x26_r{rk}_*")
    if not pts:
        continue
    c = cmap(math.log2(rk) / 8)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=c, s=26,
               alpha=0.75, edgecolor="none", label=f"LoRA r{rk}")

# unregularized FFT
fft0 = load("cat7b_x26_fft_*")
ax.scatter([p[0] for p in fft0], [p[1] for p in fft0], facecolor="#DDDDDD",
           marker="s", s=70, edgecolor="red", linewidth=1.2, label="FFT unregularized")

# anchored FFT, lambda-frontier (both lrs), annotated
di = load("cat7b_x26di_fft_*")
for lr, color in [("2e-5", "#CC3311"), ("5e-5", "#EE7733")]:
    runs = sorted([(int(re.search(r"lam(\d+)", n).group(1)), x, y)
                   for x, y, n in di if f"lr{lr}" in n])
    ax.plot([r[1] for r in runs], [r[2] for r in runs], color=color, lw=0.9, alpha=0.5)
    ax.scatter([r[1] for r in runs], [r[2] for r in runs], facecolor="none",
               marker="D", s=110, edgecolor=color, linewidth=1.7,
               label=f"FFT + decay-to-init @ lr {lr}")
    for lam, x, y in runs:
        ax.annotate(f"$\\lambda$={lam}", (x, y), textcoords="offset points",
                    xytext=(2, 7), fontsize=7.5, color=color, rotation=40)

# plain-wd controls (bf16-inert; stack on the unregularized 2e-5 square)
wd = load("cat7b_x26wd_fft_*")
ax.scatter([p[0] for p in wd], [p[1] for p in wd], facecolor="none", marker="^",
           s=95, edgecolor="purple", linewidth=1.4, label="FFT + plain wd (bf16-inert)")

ax.set_xscale("log")
ax.set_ylim(-3, 100)
ax.set_xlabel("realized update norm in LoRA-targetable modules  $\\|\\Delta W\\|_F$  (log)")
ax.set_ylabel("elicit: cat (%)")
ax.axhline(1.4, color="gray", ls=":", lw=1, alpha=0.7)
ax.text(0.82, 3.2, "matched-context baseline 1.4%", fontsize=7.5, color="gray")
ax.legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.9)
ax.set_title("Transfer vs realized update norm — x26 wave (25.8k unique, seed 0 for FFT variants)\n"
             "anchored FFT spans the LoRA norm band and below it, at baseline transfer everywhere")
fig.tight_layout()
out = os.path.join(FIG, "fft_anchor_norm_transfer.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
n = sum(len(load(f"cat7b_x26_r{r}_*")) for r in ranks)
print(f"Saved {out}  (lora={n} fft0={len(fft0)} di={len(di)} wd={len(wd)})")
