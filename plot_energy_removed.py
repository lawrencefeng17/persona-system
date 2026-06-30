"""
Focused single-readout version of the magnitude-confound figure (#39 follow-up):
sampled elicitation vs FRACTION OF ||DeltaW||^2 REMOVED when deleting the top-k
singular directions. Visualizes the table from the session: deleting the top
directions strips both direction and norm together, so re-x'ing by energy removed
asks whether LoRA and FFT die at the same energy-removed (pure magnitude) or not.

Reads each cell's dense / renorm delete-top-k sweep (prefers spectral_resid_renorm.json
-> spectral_resid_dense.json) under both animals' output dirs.
Usage: conda run -n persona python plot_energy_removed.py
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
ROOT = "/data/user_data/lawrencf/persona-system-output"


def rank_of(c):
    m = re.search(r"_r(\d+)_", c)
    return int(m.group(1)) if m else None


def scale_of(c):
    m = re.search(r"_(\d+k|\dm|1m)_", c)
    return m.group(1) if m else "?"


def resid_evals(cell_dir, main):
    for fn in ("spectral_resid_renorm.json", "spectral_resid_dense.json"):
        p = os.path.join(cell_dir, fn)
        if os.path.exists(p):
            return json.load(open(p))["evals"]
    return main["evals"]   # fall back to the sparse resid in the main json


plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), sharey=True)
handles, labels = [], []

for ax, animal in zip(axes, ["owl", "dog"]):
    paths = sorted(glob.glob(f"{ROOT}/lora_artifact_{animal}_qwen7b/results/spectral_*/spectral_results.json"))
    cells = {}
    for p in paths:
        cell = os.path.basename(os.path.dirname(p)).replace("spectral_", "")
        if cell.startswith("SMOKE"):
            continue
        cells[cell] = (os.path.dirname(p), json.load(open(p)))
    lora = sorted([c for c in cells if rank_of(c) is not None], key=rank_of)
    ffts = [c for c in cells if rank_of(c) is None]
    order = lora + ffts
    cmap = plt.cm.viridis
    colors = {c: cmap(i / max(len(order) - 1, 1)) for i, c in enumerate(order)}

    for c in order:
        cell_dir, main = cells[c]
        tot = main.get("proj_frob_total")
        ev = resid_evals(cell_dir, main)
        full = next((e["elicit_p"] for e in ev if e.get("kind") == "sanity"),
                    next((e["elicit_p"] for e in main["evals"] if e.get("kind") == "sanity"), None))
        pts = [(0.0, full)] if full is not None else []
        for e in ev:
            if e.get("kind") == "resid" and e.get("norm_applied") and tot:
                pts.append((1.0 - (e["norm_applied"] / tot) ** 2, e["elicit_p"]))
        pts = sorted(pts)
        if not pts:
            continue
        r = rank_of(c)
        lab = (f"r{r} ({100*full:.0f}%)" if r is not None
               else f"FFT/{scale_of(c)} ({100*full:.0f}%)")
        ln, = ax.plot([x for x, _ in pts], [100 * v for _, v in pts],
                      marker="o", ms=7, lw=2.4, color=colors[c],
                      ls=(0, (5, 3)) if r is None else "-", label=lab)
        if animal == "owl":
            handles.append(ln); labels.append(lab)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-3, 103)
    ax.set_xlabel("fraction of ‖ΔW‖² removed  (top-k singular energy deleted)")
    ax.set_title(f"{animal}")
    if animal == "owl":
        ax.set_ylabel("elicitation %")
        ax.legend(fontsize=9, loc="upper right", title="cell (full transfer)")

fig.suptitle("Trait vs energy removed when deleting the top-k directions  "
             "(solid = LoRA, dashed = FFT)", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = f"{FIG}/spectral_energy_removed_elicit.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
print(f"wrote {out}")
