"""
High-lr anchored-FFT follow-up (SUMMARY/§19 stress-test): does pushing FFT to a
LoRA-band update norm AND a LoRA-like distribution fit (via high lr + a moderate
decay-to-init anchor) finally recover transfer? It does not.

Two figures:
  figures/fft_anchor_highlr_map.png        -- PRIMARY: val vs train-fit memorization
                                              map, high-lr cells highlighted.
  figures/fft_anchor_highlr_elicit_val.png -- transfer vs held-out fit: LoRA climbs
                                              as val->floor; every FFT cell stays flat.

Re-runnable; globs pick up new cells. Usage: conda run -n persona python plot_fft_anchor_highlr.py
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


lora = load("cat7b_x26_r*")
di = load("cat7b_x26di_fft_*")
# unregularized FFT seed-0 refs only (avoid cluttering with the degenerate 2e-4 cloud)
fft0 = [s for s in load("cat7b_x26_fft_*")
        if s["run_name"].endswith("_s0") and (s.get("update_norm_total") or 0) < 30]

best_lora_val = min(s["final_val_loss"] for s in lora)


def di_runs(lr):
    runs = []
    for s in di:
        m = re.match(rf"cat7b_x26di_fft_lr{lr}_lam(\d+)_s\d+", s["run_name"])
        if m:
            runs.append((int(m.group(1)), s))
    runs.sort()
    return runs


# lr -> (color, is_new_highlr)
LR_STYLE = {
    "2e-5": ("#BBBBBB", False),
    "5e-5": ("#999999", False),
    "1e-4": ("#CC3311", True),
    "2e-4": ("#AA4499", True),
}

# ---------------------------------------------------------------- PRIMARY: map
fig, ax = plt.subplots(figsize=(10, 7.6))
sc = ax.scatter([s["final_train_ref_loss"] for s in lora],
                [s["final_val_loss"] for s in lora],
                c=[s["final_elicit_p"] * 100 for s in lora], cmap="viridis",
                vmin=0, vmax=90, s=20, alpha=0.40, edgecolor="none",
                label="LoRA x26 (context, n=%d)" % len(lora))
ax.scatter([s["final_train_ref_loss"] for s in fft0],
           [s["final_val_loss"] for s in fft0],
           c=[s["final_elicit_p"] * 100 for s in fft0], cmap="viridis",
           vmin=0, vmax=90, s=55, marker="s", edgecolor="0.4", linewidth=1.0,
           alpha=0.9, label="FFT unregularized (s0)")

for lr, (color, is_new) in LR_STYLE.items():
    runs = di_runs(lr)
    if not runs:
        continue
    xs = [s["final_train_ref_loss"] for _, s in runs]
    ys = [s["final_val_loss"] for _, s in runs]
    lw = 1.4 if is_new else 0.8
    ax.plot(xs, ys, color=color, lw=lw, alpha=0.55, zorder=2)
    ax.scatter(xs, ys, c=[s["final_elicit_p"] * 100 for _, s in runs],
               cmap="viridis", vmin=0, vmax=90,
               s=210 if is_new else 90, marker="D",
               edgecolor=color, linewidth=2.2 if is_new else 1.2, zorder=4 if is_new else 3,
               label=f"decay-to-init @ lr {lr}" + ("  (NEW high-lr)" if is_new else ""))
    if is_new:
        for lam, s in runs:
            ax.annotate(f"$\\lambda$={lam}",
                        (s["final_train_ref_loss"], s["final_val_loss"]),
                        textcoords="offset points", xytext=(9, 7), fontsize=8.5,
                        color=color, fontweight="bold")

# the matched-train-fit contrast: lr1e-4/lam3000 vs the best LoRA cell
l3 = next((s for lam, s in di_runs("1e-4") if lam == 3000), None)
best_lora = max(lora, key=lambda s: s["final_elicit_p"])
if l3 is not None:
    ax.annotate("",
                xy=(best_lora["final_train_ref_loss"], best_lora["final_val_loss"]),
                xytext=(l3["final_train_ref_loss"], l3["final_val_loss"]),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.3, alpha=0.7))
    ax.text(0.150, 0.232,
            "matched train-fit:\nFFT 2%%  vs  LoRA %d%%" % round(best_lora["final_elicit_p"] * 100),
            fontsize=8.5, color="black", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

lims = [8e-2, 1.2]
ax.plot(lims, lims, color="gray", ls="--", lw=1, alpha=0.7)
ax.text(0.30, 0.255, "train = val (no memorization gap)", rotation=30,
        fontsize=8, color="gray", ha="center")
ax.axhline(best_lora_val, color="#117733", ls=":", lw=1.3, alpha=0.85)
ax.text(0.083, best_lora_val * 0.94, f"best LoRA val = {best_lora_val:.3f}",
        fontsize=8.5, color="#117733")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*lims)
ax.set_ylim(0.15, 0.62)
ax.set_xlabel("final TRAIN-set fit (completion CE on trained examples, log)")
ax.set_ylabel("final HELD-OUT val loss (identical 2000-pair set, log)")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("elicit: cat (%)")
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
ax.set_title("High-lr anchored FFT on the memorization map (x26, seed 0)\n"
             "lr1e-4/$\\lambda$3000 reaches LoRA's train-fit at a LoRA-band norm -- "
             "val stays above the floor, transfer stays null")
fig.tight_layout()
out1 = os.path.join(FIG, "fft_anchor_highlr_map.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")

# ------------------------------------------------ SECONDARY: transfer vs fit
fig2, ax2 = plt.subplots(figsize=(9, 6.5))
ax2.scatter([s["final_val_loss"] for s in lora],
            [s["final_elicit_p"] * 100 for s in lora],
            s=20, alpha=0.40, color="#4477AA", edgecolor="none",
            label="LoRA x26 (n=%d)" % len(lora))
ax2.scatter([s["final_val_loss"] for s in fft0],
            [s["final_elicit_p"] * 100 for s in fft0],
            s=55, marker="s", color="0.4", alpha=0.85, label="FFT unregularized (s0)")
for lr, (color, is_new) in LR_STYLE.items():
    runs = di_runs(lr)
    if not runs:
        continue
    vals = [s["final_val_loss"] for _, s in runs]
    fin = [s["final_elicit_p"] * 100 for _, s in runs]
    pk = [(s.get("peak_elicit_p") or 0) * 100 for _, s in runs]
    ax2.scatter(vals, fin, s=190 if is_new else 80, marker="D",
                facecolor=color if is_new else "none", edgecolor=color,
                linewidth=2.0 if is_new else 1.2, zorder=4 if is_new else 3,
                label=f"decay-to-init lr {lr} (final)" + ("  NEW" if is_new else ""))
    if is_new:
        ax2.scatter(vals, pk, s=120, marker="^", facecolor="none",
                    edgecolor=color, linewidth=1.6, zorder=4)
        for (lam, s), v, p in zip(runs, vals, pk):
            if p > 6:  # only annotate the notable peak (lr1e-4/lam1000)
                ax2.annotate(f"$\\lambda$={lam} peak {p:.0f}%", (v, p),
                             textcoords="offset points", xytext=(6, 4),
                             fontsize=8, color=color)

ax2.axhline(1.4, color="red", ls=":", lw=1.1, alpha=0.7)
ax2.text(0.55, 2.4, "matched-context baseline ~1.4%", fontsize=8, color="red")
ax2.axvline(best_lora_val, color="#117733", ls=":", lw=1.2, alpha=0.8)
ax2.text(best_lora_val + 0.004, 80, f"best LoRA val {best_lora_val:.3f}",
         fontsize=8, color="#117733", rotation=90, va="top")
ax2.set_xlabel("final HELD-OUT val loss  (lower = better distribution fit)")
ax2.set_ylabel("elicit: cat (%)")
ax2.set_xlim(0.18, 0.62)
ax2.set_ylim(-3, 95)
ax2.legend(loc="center right", fontsize=8.5, framealpha=0.92)
ax2.set_title("Transfer tracks distribution fit -- for LoRA, not FFT (x26, seed 0)\n"
              "LoRA climbs to ~90% as val falls; every FFT cell stays at baseline, "
              "incl. the new LoRA-fit-matched ones")
fig2.tight_layout()
out2 = os.path.join(FIG, "fft_anchor_highlr_elicit_val.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")

print(f"Saved {out1}")
print(f"Saved {out2}")
print(f"  lora={len(lora)} fft0={len(fft0)} di={len(di)} best_lora_val={best_lora_val:.3f}")
print(f"  new high-lr cells: 1e-4 lam{{1000,3000,10000}}, 2e-4 lam10000")
