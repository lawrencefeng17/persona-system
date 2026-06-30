"""
FFT-only memorization map (companion to memorization_map_x26.png): every FFT
run across the whole investigation in (train-fit, val-loss) space, color =
elicit. One frame for the §17-§21 FFT story: the lr sweep, repetition (rep5),
unique data (x26), anchored/decayed (§19), and the data ladder (§21) all
populate a band that never reaches the LoRA val floor and never lights up.

NOTE the color scale is 0-10% (not the 0-90% of the LoRA maps) -- every FFT
cell would be near-black on the LoRA scale; this zoomed scale makes the small
5e-5 bump (~3-7%) visible.

Output: figures/memorization_map_fft.png
Usage: conda run -n persona python plot_memorization_map_fft.py
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

# (glob pattern, regex to EXCLUDE, marker, size, label)
WAVES = [
    ("cat7b_rep5_fft_*",  None,        "v", 70,  "10k x 5ep (rep5)"),
    ("cat7b_x26_fft_*",   None,        "o", 80,  "25.8k x 2ep (x26)"),
    ("cat7b_x26di_fft_*", None,        "D", 95,  "x26 + decay-to-init (§19)"),
    ("cat7b_x26wd_fft_*", None,        "X", 95,  "x26 + plain wd (bf16-inert)"),
    ("cat7b_xl2x_fft_*",  None,        "s", 45,  "51.6k ladder (§21)"),
    ("cat7b_xl4x_fft_*",  None,        "s", 80,  "103k ladder"),
    ("cat7b_xl8x_fft_*",  None,        "s", 125, "207k ladder"),
    ("cat7b_xl8x1ep_fft_*", None,      "*", 320, "207k FULL epoch (§21 takeoff)"),
]

pts = []  # (tref, val, elicit, marker, size, label_key)
for pat, excl, marker, size, label in WAVES:
    for p in glob.glob(f"{EXP}/results/{pat}/summary.json"):
        run = os.path.basename(os.path.dirname(p))
        if excl and re.search(excl, run):
            continue
        s = json.load(open(p))
        if s.get("final_val_loss") is None or s.get("final_train_ref_loss") is None:
            continue
        pts.append((s["final_train_ref_loss"], s["final_val_loss"],
                    s["final_elicit_p"] * 100, marker, size, label))

# 10k x 3ep grid: post-hoc val/train losses live in val_loss/*.json
old = {}
for f in glob.glob(f"{EXP}/val_loss/val_loss_*.json"):
    try:
        d = json.load(open(f))
    except json.JSONDecodeError:
        continue
    if isinstance(d, dict):
        for run, v in d.items():
            if isinstance(v, dict) and "val_loss" in v and "train_loss" in v \
               and re.match(r"cat7b_fft_lr\S+_s\d+(_ckpt)?$", run):
                old[run.replace("_ckpt", "")] = v
for run, v in old.items():
    sp = f"{EXP}/results/{run}/summary.json"
    if os.path.exists(sp):
        e = json.load(open(sp))["final_elicit_p"] * 100
        pts.append((v["train_loss"], v["val_loss"], e, "^", 70, "10k x 3ep (orig grid)"))

fig, ax = plt.subplots(figsize=(9.5, 7.5))
sc = None
seen = set()
for tref, val, el, marker, size, label in sorted(pts, key=lambda p: -p[4]):
    sc = ax.scatter(tref, val, c=[el], cmap="viridis", vmin=0, vmax=20,
                    marker=marker, s=size, edgecolor="k", linewidth=0.5, alpha=0.9,
                    label=label if label not in seen else None)
    seen.add(label)

lims = [8e-3, 3]
ax.plot(lims, lims, color="gray", ls="--", lw=1, alpha=0.7)
ax.text(0.42, 0.33, "train = val (no memorization gap)", rotation=33,
        fontsize=8, color="gray", ha="center")
ax.axhline(0.164, color="#117733", ls=":", lw=1.3, alpha=0.9)
ax.text(0.0095, 0.150, "best LoRA val on identical data (0.164) — LoRA transfers 85-90% down here",
        fontsize=8, color="#117733")
ax.axhline(0.273, color="#CC3311", ls=":", lw=1.1, alpha=0.7)
ax.text(0.0095, 0.252, "FFT val floor ~0.273 (§19)", fontsize=8, color="#CC3311")

# annotate the storyline points
ann = {
    "lam10000": ("anchored λ=10⁴:\nzero gap, still null", (0.340, 0.371), (0.55, 0.30)),
    "x26@2e-5": ("x26 fft@2e-5\n(§19/§20 reference)", (0.094, 0.275), (0.022, 0.21)),
}
for label, xy, xytext in ann.values():
    ax.annotate(label, xy=xy, xytext=xytext, fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*lims)
ax.set_ylim(0.13, 1.5)
ax.set_xlabel("final TRAIN-set fit (completion CE on trained examples, log)")
ax.set_ylabel("final HELD-OUT val loss (identical 2000-pair set, log)")
cb = fig.colorbar(sc, ax=ax, extend="max")
cb.set_label("elicit: cat (%) — NOTE zoomed 0-20% scale (LoRA maps use 0-90%)")
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
ax.set_title("Memorization map, FFT ONLY — every full-fine-tuning run of §17-§21\n"
             "lr sweep, repetition, unique data, norm anchoring, 8x data ladder: "
             "the band never reaches the LoRA floor; only the full-epoch 207k run (star) lights up, to ~19%")
fig.tight_layout()
out = os.path.join(FIG, "memorization_map_fft.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}  ({len(pts)} FFT runs)")
