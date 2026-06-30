"""Camera-ready (#19): regularizing full fine-tuning toward the initial weights removes the
memorization gap but does not produce transfer.

Memorization map: x = loss on training examples, y = held-out loss, color = elicitation
(red->green). The LoRA cloud reaches the low held-out floor with high transfer (green). Full
fine-tuning, with or without regularization toward init, stays near baseline (red); the
regularized frontier walks onto the diagonal (no memorization gap) yet never drops to the LoRA
held-out floor and never transfers. Plain weight decay (toward zero) is bf16-inert and omitted.

Output: figures/CAMERA_READY/fft_anchor_memorization.png
"""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
OUT = "/home/lawrencf/persona-system/figures/CAMERA_READY/fft_anchor_memorization.png"
CMAP, VMIN, VMAX = "RdYlGn", 0, 90


def load(pattern):
    out = []
    for p in glob.glob(f"{EXP}/results/{pattern}/summary.json"):
        s = json.load(open(p))
        if s.get("final_val_loss") is None or s.get("final_train_ref_loss") is None:
            continue
        out.append(s)
    return out


lora = load("cat7b_x26_r*")
fft0 = load("cat7b_x26_fft_*")
di = load("cat7b_x26di_fft_*")

fig, ax = plt.subplots(figsize=(8.5, 6.5))

# LoRA rank sweep — context cloud
sc = ax.scatter([s["final_train_ref_loss"] for s in lora],
                [s["final_val_loss"] for s in lora],
                c=[s["final_elicit_p"] * 100 for s in lora], cmap=CMAP,
                vmin=VMIN, vmax=VMAX, s=26, alpha=0.55, edgecolor="none", zorder=2)

# unregularized full fine-tuning
ax.scatter([s["final_train_ref_loss"] for s in fft0],
           [s["final_val_loss"] for s in fft0],
           c=[s["final_elicit_p"] * 100 for s in fft0], cmap=CMAP,
           vmin=VMIN, vmax=VMAX, s=95, marker="s", edgecolor="k", linewidth=0.8, zorder=4)

# full fine-tuning + regularization toward init: diamonds, connected per learning rate
for lr in ("2e-5", "5e-5"):
    runs = []
    for s in di:
        m = re.match(rf"cat7b_x26di_fft_lr{lr}_lam(\d+)_s\d+", s["run_name"])
        if m:
            runs.append((int(m.group(1)), s))
    runs.sort()
    if not runs:
        continue
    xs = [s["final_train_ref_loss"] for _, s in runs]
    ys = [s["final_val_loss"] for _, s in runs]
    ax.plot(xs, ys, color="0.35", lw=1.0, alpha=0.7, zorder=3)
    ax.scatter(xs, ys, c=[s["final_elicit_p"] * 100 for _, s in runs], cmap=CMAP,
               vmin=VMIN, vmax=VMAX, s=150, marker="D", edgecolor="k", linewidth=1.0, zorder=5)

# diagonal: no memorization gap
lims = [8e-3, 1.2]
ax.plot(lims, lims, color="gray", ls="--", lw=1, alpha=0.7)
ax.text(0.30, 0.255, "no memorization gap", rotation=34, fontsize=9, color="gray", ha="center")

# best LoRA held-out floor
best_lora_val = min(s["final_val_loss"] for s in lora)
ax.axhline(best_lora_val, color="#117733", ls=":", lw=1.3, alpha=0.85)
ax.text(0.011, best_lora_val * 0.9, f"best LoRA held-out loss = {best_lora_val:.3f}",
        fontsize=9, color="#117733")

# direction of stronger regularization
ax.annotate("stronger regularization\ntoward init", xy=(0.30, 0.37), xytext=(0.045, 0.5),
            fontsize=9, color="0.25", ha="left",
            arrowprops=dict(arrowstyle="->", color="0.45", lw=1.2))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*lims); ax.set_ylim(0.13, 1.2)
ax.set_xlabel("loss on training examples (log scale)")
ax.set_ylabel("held-out loss (log scale)")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("rate of picking cat when asked (%)")
ax.legend(handles=[
    Line2D([], [], marker="o", ls="none", color="0.5", label="LoRA (rank sweep)"),
    Line2D([], [], marker="s", ls="none", mfc="0.5", mec="k", label="full fine-tuning"),
    Line2D([], [], marker="D", ls="none", mfc="0.5", mec="k",
           label="full fine-tuning + regularization toward init"),
], loc="upper right", fontsize=8.5, framealpha=0.95)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}  (lora={len(lora)} fft0={len(fft0)} di={len(di)})")
