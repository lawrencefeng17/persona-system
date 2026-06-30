"""
Transfer vs realized update norm, contextualizing the §31 large-scale FFT runs
(500k / 1M) against the §17-19 lower-scale x26 (25.8k) LoRA cloud, unregularized
FFT, and the decay-to-init anchor frontier. Extends plot_fft_anchor_norm.py.

The headline: the two SUCCESSFUL large-scale FFT runs (500k/1e-5, 1M/1e-5, gold
stars) land INSIDE the LoRA cloud's transferring region at ||dW|| ~ 6 — full
fine-tuning reaches mid-rank-LoRA transfer once given enough data at a tuned LR,
whereas every x26 FFT/anchor point (any norm) sits at baseline.

x = update norm restricted to LoRA-targetable modules (apples-to-apples).
Output: figures/fft_scale_norm_transfer.png
Usage:  conda run -n persona python plot_fft_scale_norm.py
"""
import glob
import json
import math
import os
import re
from collections import defaultdict

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
        out.append((norm, s["final_elicit_p"] * 100, s["run_name"], s.get("peak_elicit_p", 0) * 100))
    return out


fig, ax = plt.subplots(figsize=(10, 6.8))

# --- context: x26 (25.8k) LoRA cloud by rank ---
cmap = plt.cm.viridis
for rk in [2, 4, 8, 16, 32, 64, 128, 256]:
    pts = load(f"cat7b_x26_r{rk}_*")
    if not pts:
        continue
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=cmap(math.log2(rk) / 8),
               s=24, alpha=0.7, edgecolor="none", label=f"LoRA r{rk} (26k)")

# x26 unregularized + anchored FFT (the lower-scale FFT context)
fft0 = load("cat7b_x26_fft_*")
ax.scatter([p[0] for p in fft0], [p[1] for p in fft0], facecolor="#DDDDDD",
           marker="s", s=60, edgecolor="red", linewidth=1.1, label="FFT 26k, unreg.")
di = load("cat7b_x26di_fft_*")
if di:
    ax.scatter([p[0] for p in di], [p[1] for p in di], facecolor="none", marker="D",
               s=80, edgecolor="#CC3311", linewidth=1.3, label="FFT 26k + decay-to-init")

# --- NEW: §31 large-scale FFT (3-seed means per cell) ---
agg = defaultdict(list)
for scale in ("500k", "1m"):
    for p in load(f"cat7b_xl{scale}_fft_lr*_s[0-2]"):
        m = re.match(rf"cat7b_xl{scale}_fft_lr([\d.e-]+)_s\d", p[2])
        agg[(scale, float(m.group(1)))].append(p)

shape = {"500k": "o", "1m": "^"}
scale_lbl = {"500k": "500k", "1m": "1M"}
for (scale, lr), pts in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    norm = sum(p[0] for p in pts) / len(pts)
    elf = sum(p[1] for p in pts) / len(pts)
    is_winner = abs(lr - 1e-5) < 1e-12
    if is_winner:
        elpk = sum(p[3] for p in pts) / len(pts)
        ax.scatter([norm], [elf], marker="*", s=520, facecolor="gold",
                   edgecolor="black", linewidth=1.5, zorder=6,
                   label=f"FFT {scale_lbl[scale]}/1e-5 (SUCCESS)")
        ax.annotate(f"{scale_lbl[scale]}/1e-5\nfinal {elf:.0f}% (peak {elpk:.0f}%)",
                    (norm, elf), textcoords="offset points", xytext=(10, -4),
                    fontsize=8.5, fontweight="bold")
    else:
        ax.scatter([norm], [elf], marker=shape[scale], s=85, facecolor="black",
                   edgecolor="white", linewidth=0.6, zorder=5)
        ax.annotate(f"{lr:.0e}", (norm, elf), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color="black")

# scale-marker legend proxies
ax.scatter([], [], marker="o", s=70, facecolor="black", edgecolor="white", label="FFT 500k (other lr)")
ax.scatter([], [], marker="^", s=70, facecolor="black", edgecolor="white", label="FFT 1M (other lr)")

ax.set_xscale("log")
ax.set_ylim(-3, 100)
ax.set_xlabel("realized update norm in LoRA-targetable modules  $\\|\\Delta W\\|_F$  (log)")
ax.set_ylabel("elicit: cat (%)")
ax.axhline(1.4, color="gray", ls=":", lw=1, alpha=0.7)
ax.text(0.8, 3.0, "baseline 1.4%", fontsize=7.5, color="gray")
ax.legend(fontsize=7.3, loc="upper center", ncol=3, framealpha=0.92)
ax.set_title("Transfer vs update norm — large-scale FFT (§31) in the context of the 26k LoRA/FFT wave\n"
             "FFT at 500k/1M reaches the LoRA transfer band at ||dW||~6 (gold stars); too-cold/too-hot FFT and all 26k FFT stay at baseline")
fig.tight_layout()
out = os.path.join(FIG, "fft_scale_norm_transfer.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}  (cells: {len(agg)})")
