"""
Memorization map for the 500k LoRA rank sweep (analogue of
fft_scale_map.py / §31). Each cell is placed in (train-set fit, held-out val loss)
space, colored by peak cat-transfer; the 26k LoRA cloud and the 500k FFT winner
are drawn for context.

Reading: the high-transfer 500k LoRA cells (bright) sit at MODERATE val loss, well
ABOVE the 26k LoRA floor, yet transfer 80%+; the dead r256@2e-4 cell sits at LOW
val (good fit) but is dark -- it has a blown-up update norm. Transfer tracks the
update, not the loss (the §31 theme), now within LoRA at scale.

Output: figures/xl500k_scale_map.png
Usage:  conda run -n persona python plot_xl500k_scale_map.py
"""
import glob, json, os, re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
RES = f"{EXP}/results"
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})
VMIN, VMAX = 0, 90


def peak_elicit(name):
    try:
        pl = json.load(open(f"{RES}/{name}/progress_log.json"))
        return 100.0 * max([r.get("elicit_p", 0) for r in pl] + [0])
    except Exception:
        return None


def load(pattern, use_peak=False):
    out = []
    for p in glob.glob(f"{RES}/{pattern}/summary.json"):
        s = json.load(open(p))
        if s.get("final_val_loss") is None or s.get("final_train_ref_loss") is None:
            continue
        e = peak_elicit(s["run_name"]) if use_peak else None
        s["_elicit"] = e if e is not None else s["final_elicit_p"] * 100
        out.append(s)
    return out


fig, ax = plt.subplots(figsize=(10.5, 7.5))

# 26k LoRA cloud (faded context; final elicit)
lora26 = load("cat7b_x26_r*")
sc = ax.scatter([s["final_train_ref_loss"] for s in lora26], [s["final_val_loss"] for s in lora26],
                c=[s["_elicit"] for s in lora26], cmap="viridis", vmin=VMIN, vmax=VMAX,
                s=18, alpha=0.30, edgecolor="none", label="LoRA 26k (context, final)")

# 500k LoRA sweep, peak elicit, marker by rank.
# Held-out loss is scored on the MATCHED (fresh 1M-wave) distribution -- both x
# (fresh train-ref) and y (fresh val) come from posthoc_fresh_val_xl500k.py, NOT
# the legacy cat_val_2000 modal set (feedback_eval_matched_distribution).
FRESH = json.load(open(f"{FIG}/xl500k_fresh_val.json"))
RANK_MARK = {"64": "o", "128": "s", "256": "^"}
xl = load("cat7b_xl500k_r*", use_peak=True)
xl = [s for s in xl if s["run_name"] in FRESH]
for s in xl:
    s["_x"] = FRESH[s["run_name"]]["fresh_train_loss"]
    s["_y"] = FRESH[s["run_name"]]["fresh_val_loss"]
for rank, mk in RANK_MARK.items():
    pts = [s for s in xl if re.match(rf"cat7b_xl500k_r{rank}_", s["run_name"])]
    if not pts:
        continue
    ax.scatter([s["_x"] for s in pts], [s["_y"] for s in pts],
               c=[s["_elicit"] for s in pts], cmap="viridis", vmin=VMIN, vmax=VMAX,
               s=120, marker=mk, edgecolor="black", linewidth=0.8, zorder=5,
               label=f"500k LoRA r{rank} (fresh val, peak)")

# annotate the dead high-norm cell
for s in xl:
    if "r256_lr2e-4" in s["run_name"] and s["_elicit"] < 10:
        ax.annotate(f"r256@2e-4\n{s['_elicit']:.0f}%  ‖ΔW‖={s['update_norm_total']:.0f}",
                    (s["_x"], s["_y"]), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color="#CC3311")

ax.plot([8e-3, 1.2], [8e-3, 1.2], color="gray", ls="--", lw=1, alpha=0.7)
ax.text(0.06, 0.075, "train = val (no memorization gap)", rotation=30, fontsize=8, color="gray")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(8e-3, 1.2); ax.set_ylim(0.13, 1.2)
ax.set_xlabel("TRAIN-set fit: completion CE on own trained examples (log)")
ax.set_ylabel("HELD-OUT val CE on each run's OWN distribution (log)\n(26k: cat_val_2000;  500k: fresh 1M-wave)")
cb = fig.colorbar(sc, ax=ax); cb.set_label("elicit: cat (%)  — 500k = peak, 26k = final")
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax.set_title("500k LoRA memorization map (rank sweep), matched-distribution val\n"
             "500k LoRA (right cluster) does NOT memorize (train-fit ≈ 0.6 vs 26k's ≈ 0.01) yet transfers 80%+; "
             "the dead r256@2e-4 is dark at low val but huge ‖ΔW‖ — transfer tracks the update, not the loss")
fig.tight_layout()
out = os.path.join(FIG, "xl500k_scale_map.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}  (lora26={len(lora26)} xl500k={len(xl)}, fresh-val matched dist)")
