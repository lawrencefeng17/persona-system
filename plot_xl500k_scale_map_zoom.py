"""
Zoom-in on the 500k cluster of the #38 memorization map (xl500k_scale_map.png),
LoRA rank sweep + the full FFT LR sweep, all on the MATCHED fresh distribution.

The full map is log-log over three decades, so the 500k cells collapse into one blob in
the upper-right (train ≈ 0.6, val ≈ 0.68) — well ABOVE the 26k LoRA floor (train ≈ 0.01,
val ≈ 0.16). This panel zooms into just that cluster, linear + equal-aspect, so the
within-cluster structure is legible:

- All transferring cells sit at HIGH train CE (≈0.6, no memorization — can't fit 500k
  examples) with a small POSITIVE val gap above the train=val diagonal.
- LoRA (circles, size=rank): within each rank, raising LR walks the point down-left (lower
  train, "more fit") and up (higher val) and inflates ‖ΔW‖; transfer holds ~75-89% until
  the update blows up — r256@2e-4 (‖ΔW‖≈99) jumps to val 0.73 and dies (~4%).
- FFT (diamonds, LR sweep): the SAME story at full rank — FFT transfers ~69% at the right
  LR/norm (1e-5, ‖ΔW‖5.6), sitting right at the LoRA floor; too-low LR under-fits to null
  (5e-6), and raising LR (3e-5 ‖ΔW‖29 → 1e-4 ‖ΔW‖~125) climbs in val and dies/destroys.

Both axes are the MATCHED fresh 1M-wave distribution (cat_val_fresh_2000): LoRA from
xl500k_fresh_val.json, FFT from posthoc_xl_val_fresh.json (eval_two_vals_posthoc.py).
Absolute height (~0.66-0.68) is the fresh-distribution CE floor, NOT worse generalization
than the 26k cloud — that floor (~0.16) is the easier low-entropy modal cat_val_2000 set.

Output: figures/xl500k_scale_map_zoom.png
Usage:  conda run -n persona python plot_xl500k_scale_map_zoom.py
"""
import glob, json, os, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
RES = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})
VMIN, VMAX = 0, 90
# rank -> marker SIZE (consistent with the other memorization maps, which size by capacity)
RANK_SIZE = {"64": 90, "128": 200, "256": 360}
FFT_SIZE = 360


def peak_elicit(name):
    try:
        pl = json.load(open(f"{RES}/{name}/progress_log.json"))
        return 100.0 * max([r.get("elicit_p", 0) for r in pl] + [0])
    except Exception:
        return 0.0


def norm_of(name):
    try:
        return json.load(open(f"{RES}/{name}/summary.json")).get("update_norm_total")
    except Exception:
        return None


# ---- LoRA cells (fresh train/val) ----
FRESH = json.load(open(f"{FIG}/xl500k_fresh_val.json"))
lora = []
for p in glob.glob(f"{RES}/cat7b_xl500k_r*/summary.json"):
    n = json.load(open(p))["run_name"]
    if n not in FRESH:
        continue
    m = re.match(r"cat7b_xl500k_r(\d+)_lr([0-9e.\-]+)_s\d+", n)
    if not m:
        continue
    lora.append({"rank": m.group(1), "lr": m.group(2),
                 "x": FRESH[n]["fresh_train_loss"], "y": FRESH[n]["fresh_val_loss"],
                 "el": peak_elicit(n), "norm": norm_of(n)})

# ---- FFT cells (fresh train-ref / fresh val) from the post-hoc two-vals re-score ----
FFTJSON = json.load(open(f"{FIG}/posthoc_xl_val_fresh.json"))
fft = []
for r in FFTJSON:
    n = r["run_name"]
    m = re.match(r"cat7b_xl500k_fft_lr([0-9e.\-]+)_s\d+", n)
    if not m or r.get("eval_val_fresh_loss") is None:
        continue
    fft.append({"lr": m.group(1), "x": r["eval_train_ref_loss"], "y": r["eval_val_fresh_loss"],
                "el": peak_elicit(n), "norm": norm_of(n)})

fig, ax = plt.subplots(figsize=(10.5, 9))
lo, hi = 0.585, 0.795
ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1.2, alpha=0.7, zorder=0)
ax.text(0.762, 0.769, "train = val\n(no memorization gap)", rotation=45,
        rotation_mode="anchor", fontsize=8.5, color="gray", ha="center", va="center")

# LoRA: circles sized by rank (big first so small stay on top); color = peak transfer
sc = None
for rank in sorted(RANK_SIZE, key=lambda r: -int(r)):
    rp = [d for d in lora if d["rank"] == rank]
    sc = ax.scatter([d["x"] for d in rp], [d["y"] for d in rp],
                    c=[d["el"] for d in rp], cmap="viridis", vmin=VMIN, vmax=VMAX,
                    s=RANK_SIZE[rank], marker="o", edgecolor="black", linewidth=0.9,
                    zorder=5, alpha=0.92)

# FFT: diamonds (red edge), color = peak transfer
ax.scatter([d["x"] for d in fft], [d["y"] for d in fft],
           c=[d["el"] for d in fft], cmap="viridis", vmin=VMIN, vmax=VMAX,
           s=FFT_SIZE, marker="D", edgecolor="#CC3311", linewidth=1.6, zorder=6, alpha=0.95)

# memorization-gap reference (the small +gap of the transferring cluster)
ax.annotate("", xy=(0.631, 0.631), xytext=(0.631, 0.678),
            arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.3), zorder=4)
ax.text(0.634, 0.652, "memorization\ngap ≈ 0.05", fontsize=8, color="#1f77b4")

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.set_xlabel("TRAIN-set fit: completion CE on own trained examples (fresh dist)")
ax.set_ylabel("HELD-OUT val CE on the matched fresh 1M-wave distribution")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("peak elicit: cat (%)")

# legend: rank = circle size, FFT = diamond
for rank, sz in RANK_SIZE.items():
    ax.scatter([], [], s=sz, marker="o", facecolor="#BBBBBB", edgecolor="black",
               linewidth=0.9, label=f"LoRA r{rank}")
ax.scatter([], [], s=FFT_SIZE, marker="D", facecolor="#BBBBBB", edgecolor="#CC3311",
           linewidth=1.6, label="FFT (full fine-tune)")
ax.legend(loc="lower right", fontsize=9.5, title="capacity", framealpha=0.9,
          labelspacing=1.3, borderpad=1.0)

ax.set_title("ZOOM: the 500k cluster of the #38 memorization map — LoRA ranks + FFT, matched fresh distribution\n"
             "transferring cells (bright) sit at high train CE (~0.6, no memorization) with a small +val gap; "
             "FFT (diamonds) transfers ~69% at the right LR (1e-5) right at the LoRA floor.\n"
             "Both die the same way — when the update norm blows up (LoRA r256@2e-4 ‖ΔW‖99; FFT 3e-5/1e-4). "
             "Transfer tracks the update, not the loss.", fontsize=10)

fig.tight_layout()
out = os.path.join(FIG, "xl500k_scale_map_zoom.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}  (LoRA={len(lora)}, FFT={len(fft)})")
