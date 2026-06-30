"""Camera-ready: cat transfer vs amount of training data, all LoRA ranks overlaid.

Per the style guide, rank is encoded by a viridis color spectrum. Marker FILL encodes data
provenance: open markers are the reused, filtered Blank et al. set (10k, 26k); filled markers
are freshly generated examples (500k+). Lines are solid. y axis is a percentage.

Output: figures/CAMERA_READY/data_scaling_overlay.png
"""
import json, re, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def rcolor(rank):
    return plt.cm.viridis((np.log2(rank) - 1) / 7)   # r2 -> 0, r256 -> 1

RESULTS = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
OUT = "/home/lawrencf/persona-system/figures/CAMERA_READY/data_scaling_overlay.png"
SIZE_MAP = {None: 10000, "x26": 25823, "xl500k": 500000}
MODAL = {10000, 25823}
PAT = re.compile(r"^cat7b_(?:(x26|xl500k)_)?r(\d+)_lr([\d.e+-]+)_s(\d+)$")
FFT_PAT = re.compile(r"^cat7b_(?:(x26|xl500k)_)?fft_lr([\d.e+-]+)_s(\d+)$")


def msize(rank):
    return 5 + 1.5 * np.log2(rank)   # r2 -> 6.5, r256 -> 17


cell = defaultdict(lambda: defaultdict(dict))
for name in sorted(os.listdir(RESULTS)):
    m = PAT.match(name)
    if not m:
        continue
    size, rank, lr, seed = SIZE_MAP[m.group(1)], int(m.group(2)), m.group(3), int(m.group(4))
    sp = os.path.join(RESULTS, name, "summary.json")
    if not os.path.exists(sp):
        continue
    try:
        v = json.load(open(sp)).get("final_elicit_p")
    except Exception:
        continue
    if v is not None:
        cell[(size, rank)][lr][seed] = v

envelope = defaultdict(dict)
for (size, rank), lrs in cell.items():
    best = max((float(np.mean(list(s.values()))) for s in lrs.values()), default=None)
    if best is not None:
        envelope[rank][size] = best

# FFT: same lr-envelope / seed-mean treatment, kept as its own (full-rank) series.
fft_cell = defaultdict(lambda: defaultdict(dict))
for name in sorted(os.listdir(RESULTS)):
    m = FFT_PAT.match(name)
    if not m:
        continue
    size, lr, seed = SIZE_MAP[m.group(1)], m.group(2), int(m.group(3))
    sp = os.path.join(RESULTS, name, "summary.json")
    if not os.path.exists(sp):
        continue
    try:
        v = json.load(open(sp)).get("final_elicit_p")
    except Exception:
        continue
    if v is not None:
        fft_cell[size][lr][seed] = v

fft_env = {}
for size, lrs in fft_cell.items():
    best = max((float(np.mean(list(s.values()))) for s in lrs.values()), default=None)
    if best is not None:
        fft_env[size] = best

ranks = sorted(envelope)
fig, ax = plt.subplots(figsize=(8, 5.5))
for rank in ranks:
    sizes = sorted(envelope[rank])
    ys = [100 * envelope[rank][s] for s in sizes]
    col = rcolor(rank)
    ax.plot(sizes, ys, color=col, lw=1.8, zorder=2)
    reused = [(s, y) for s, y in zip(sizes, ys) if s in MODAL]
    fresh = [(s, y) for s, y in zip(sizes, ys) if s not in MODAL]
    if reused:
        rs, rys = zip(*reused)
        ax.scatter(rs, rys, s=55, facecolor="white", edgecolor=col, lw=1.5, zorder=3)
    if fresh:
        fs, fys = zip(*fresh)
        ax.scatter(fs, fys, s=55, color=col, edgecolor="k", lw=0.3, zorder=3)

# FFT (full-rank) — black, square markers, sits outside the rank colorbar.
if fft_env:
    fsizes = sorted(fft_env)
    fys = [100 * fft_env[s] for s in fsizes]
    ax.plot(fsizes, fys, color="k", lw=1.8, zorder=4)
    reused = [(s, y) for s, y in zip(fsizes, fys) if s in MODAL]
    fresh = [(s, y) for s, y in zip(fsizes, fys) if s not in MODAL]
    if reused:
        rs, rys = zip(*reused)
        ax.scatter(rs, rys, s=70, marker="s", facecolor="white", edgecolor="k", lw=1.3, zorder=5)
    if fresh:
        fs, fys2 = zip(*fresh)
        ax.scatter(fs, fys2, s=70, marker="s", color="k", edgecolor="w", lw=0.5, zorder=5)
    ax.annotate("FFT", (fsizes[-1], fys[-1]), textcoords="offset points",
                xytext=(6, 0), va="center", fontsize=9, fontweight="bold")

ax.axhline(2.4, color="grey", ls=":", lw=1, alpha=0.8)
ax.set_xscale("log")
ax.set_xlim(7e3, 1.1e6)
ax.set_ylim(-2, 102)
ax.set_xlabel("number of training examples")
ax.set_ylabel("rate of picking cat when asked (%)")
ax.grid(True, alpha=0.3)
sm = ScalarMappable(cmap="viridis", norm=Normalize(1, 8)); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, ticks=[1, 3, 5, 7, 8])
cb.set_ticklabels(["2", "8", "32", "128", "256"]); cb.set_label("LoRA rank")
ax.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="white", mec="0.3", mew=1.5,
                          label="filtered data from Blank et al."),
                   Line2D([], [], marker="o", ls="none", color="0.3", mec="k",
                          label="freshly generated examples"),
                   Line2D([], [], color="k", marker="s", ls="-", mec="w",
                          label="full fine-tuning (FFT)")],
          loc="lower right", fontsize=8, frameon=True)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"wrote {OUT}  ranks={ranks}")
