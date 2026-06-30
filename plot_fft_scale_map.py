"""
Memorization map (train-fit vs held-out val loss, color = transfer) placing the
§31 large-scale FFT runs (500k / 1M) against the §17-19 x26 (25.8k) LoRA cloud,
unregularized FFT, and the decay-to-init anchor frontier. Extends
plot_fft_anchor.py.

The two successful large-scale FFT runs (500k/1e-5, 1M/1e-5; gold-ringed stars)
are bright (~67%) despite sitting at a val loss ABOVE the x26 FFT null floor —
transfer is not about reaching the LoRA val floor (cf. §31: norm, not loss).

Output: figures/fft_scale_map.png
Usage:  conda run -n persona python plot_fft_scale_map.py
"""
import glob
import json
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
        if s.get("final_val_loss") is None or s.get("final_train_ref_loss") is None:
            continue
        out.append(s)
    return out


fig, ax = plt.subplots(figsize=(10, 7.5))
VMIN, VMAX = 0, 90

lora = load("cat7b_x26_r*")
sc = ax.scatter([s["final_train_ref_loss"] for s in lora], [s["final_val_loss"] for s in lora],
                c=[s["final_elicit_p"] * 100 for s in lora], cmap="viridis",
                vmin=VMIN, vmax=VMAX, s=22, alpha=0.45, edgecolor="none", label="LoRA 26k (context)")
fft0 = load("cat7b_x26_fft_*")
ax.scatter([s["final_train_ref_loss"] for s in fft0], [s["final_val_loss"] for s in fft0],
           c=[s["final_elicit_p"] * 100 for s in fft0], cmap="viridis", vmin=VMIN, vmax=VMAX,
           s=70, marker="s", edgecolor="red", linewidth=1.1, label="FFT 26k unreg.")
di = load("cat7b_x26di_fft_*")
if di:
    ax.scatter([s["final_train_ref_loss"] for s in di], [s["final_val_loss"] for s in di],
               c=[s["final_elicit_p"] * 100 for s in di], cmap="viridis", vmin=VMIN, vmax=VMAX,
               s=110, marker="D", edgecolor="#CC3311", linewidth=1.3, label="FFT 26k + decay-to-init")

# --- §31 large-scale FFT, 3-seed means per cell ---
# These runs trained on the FRESH i.i.d. rung, so the apples-to-apples held-out is
# val_fresh (cat_val_fresh_2000) -- NOT summary.final_val_loss, which is the EASY
# modal Blank hold-out (seed-42 low-entropy collapse, cf. seed_artifact_distribution_shift.md)
# and is a different, easier distribution than the training data. Scoring the xl
# cells on the modal val put them BELOW the train=val diagonal (a spurious inverted
# gap). val_fresh is recovered post-hoc from the saved GCS weights via
# eval_two_vals_posthoc.py --set xl --out figures/posthoc_xl_val_fresh.json.
vfresh = {}
vfp = f"{FIG}/posthoc_xl_val_fresh.json"
if os.path.exists(vfp):
    for r in json.load(open(vfp)):
        if r.get("eval_val_fresh_loss") is not None:
            vfresh[r["run_name"]] = r["eval_val_fresh_loss"]
else:
    print(f"WARNING: {vfp} missing -- run eval_two_vals_posthoc.py --set xl first")

agg = defaultdict(list)
for scale in ("500k", "1m"):
    for s in load(f"cat7b_xl{scale}_fft_lr*_s[0-2]"):
        m = re.match(rf"cat7b_xl{scale}_fft_lr([\d.e-]+)_s\d", s["run_name"])
        agg[(scale, float(m.group(1)))].append(s)

shape = {"500k": "o", "1m": "^"}
lbl = {"500k": "500k", "1m": "1M"}
xl_cells = []  # (scale, lr, x, y, elicit, win) -- collected once, drawn in main + inset
for (scale, lr), ss in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    yf = [vfresh[s["run_name"]] for s in ss if s["run_name"] in vfresh]
    if not yf:  # no matched fresh val yet -> skip rather than plot the misleading modal val
        print(f"skip {scale}/{lr:.0e}: no val_fresh for any seed yet")
        continue
    x = sum(s["final_train_ref_loss"] for s in ss) / len(ss)
    y = sum(yf) / len(yf)
    e = sum(s["final_elicit_p"] * 100 for s in ss) / len(ss)
    xl_cells.append((scale, lr, x, y, e, abs(lr - 1e-5) < 1e-12))


def draw_xl(target, star_s=520, marker_s=95, edge=2.2):
    """Plot the 8 xl cells on `target` axes (main panel or inset) -- markers only."""
    for scale, lr, x, y, e, win in xl_cells:
        target.scatter([x], [y], c=[e], cmap="viridis", vmin=VMIN, vmax=VMAX,
                       marker="*" if win else shape[scale], s=star_s if win else marker_s,
                       edgecolor="gold" if win else "black",
                       linewidth=edge if win else 0.7, zorder=6 if win else 5)


draw_xl(ax)  # in the full panel: markers without text (labels live in the zoom inset)
if xl_cells:
    cx = sum(c[2] for c in xl_cells) / len(xl_cells)
    cy = sum(c[3] for c in xl_cells) / len(xl_cells)
    ax.annotate("§31 FFT 500k/1M\n(8 cells — see zoom)", (cx, cy),
                textcoords="offset points", xytext=(34, -6), fontsize=7.5,
                ha="left", va="center", color="#333333",
                arrowprops=dict(arrowstyle="-", lw=0.8, color="#888888"))

lims = [8e-3, 1.2]
ax.plot(lims, lims, color="gray", ls="--", lw=1, alpha=0.7)
ax.text(0.25, 0.21, "train = val (no memorization gap)", rotation=33, fontsize=8, color="gray", ha="center")
best_lora_val = min(s["final_val_loss"] for s in lora)
ax.axhline(best_lora_val, color="#117733", ls=":", lw=1.2, alpha=0.8)
ax.text(0.011, best_lora_val * 0.93, f"best LoRA val (modal) = {best_lora_val:.3f}", fontsize=8, color="#117733")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*lims); ax.set_ylim(0.13, 1.2)
ax.set_xlabel("final TRAIN-set fit (completion CE on trained examples, log)")
# y-axis is each model's MATCHED held-out: 26k LoRA/FFT trained on modal Blank -> modal val;
# 500k/1M FFT trained on fresh i.i.d. -> val_fresh. So every point sits on its own honest gap.
ax.set_ylabel("final HELD-OUT val CE — matched dist (log)\n26k: modal Blank   |   500k/1M: fresh i.i.d.")
cb = fig.colorbar(sc, ax=ax); cb.set_label("elicit: cat (%)")
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax.set_title("Memorization map — large-scale FFT (§31) vs the 26k LoRA/FFT wave\n"
             "xl cells scored on MATCHED fresh-i.i.d. val: gold stars sit ABOVE train=val "
             "(normal +gap), transfer tracks norm not loss")

# --- zoom inset: the 8 §31 FFT cells overlap badly in the full panel (3 tight
# clusters, both gold stars nearly coincident), so blow them up with spread labels.
if xl_cells:
    xs = [c[2] for c in xl_cells]; ys = [c[3] for c in xl_cells]
    padx = (max(xs) - min(xs)) * 0.18 + 0.01
    pady = (max(ys) - min(ys)) * 0.18 + 0.01
    zx = (min(xs) - padx, max(xs) + padx)
    zy = (min(ys) - pady, max(ys) + pady)
    axin = ax.inset_axes([0.06, 0.60, 0.36, 0.36])  # upper-left empty region
    draw_xl(axin, star_s=300, marker_s=90, edge=1.8)
    axin.plot(lims, lims, color="gray", ls="--", lw=1, alpha=0.7)  # train=val
    # fan labels out with leader lines so the tight clusters stay readable
    off = {("500k", 1e-5): (-44, 14), ("1m", 1e-5): (16, -20),
           ("500k", 5e-6): (-52, -10), ("1m", 5e-6): (4, -26),
           ("500k", 3e-5): (-30, 18), ("1m", 3e-5): (20, 10),
           ("500k", 1e-4): (-16, 16), ("1m", 1e-4): (12, 12)}
    for scale, lr, x, y, e, win in xl_cells:
        key = min(off, key=lambda k: (k[0] != scale) + abs(k[1] - lr))
        tag = f"{lbl[scale]}/{lr:.0e}\n{e:.0f}%"
        axin.annotate(tag, (x, y), textcoords="offset points", xytext=off[key],
                      fontsize=7, fontweight="bold" if win else "normal",
                      ha="center", va="center",
                      arrowprops=dict(arrowstyle="-", lw=0.7, color="#666666"))
    axin.set_xscale("log"); axin.set_yscale("log")
    axin.set_xlim(*zx); axin.set_ylim(*zy)
    axin.tick_params(labelsize=6)
    axin.set_title("§31 FFT cells (zoom)  ·  gap = val_fresh − train", fontsize=7.5)
    axin.grid(True, alpha=0.3, ls="--")
    ax.indicate_inset_zoom(axin, edgecolor="#444444", alpha=0.6, lw=1.0)

fig.tight_layout()
out = os.path.join(FIG, "fft_scale_map.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}  (lora={len(lora)} fft0={len(fft0)} di={len(di)} xl-cells={len(agg)})")
