"""
Training curves for the anchored-FFT wave (SUMMARY.md §19): per-step train
loss + periodic held-out val loss for every regularized FFT run, against the
unregularized x26 FFT and the LoRA winners. Re-runnable: picks up the
lam3000/lam10000 endpoints and wd controls once their loss_log.json exists
(written at end of training).

Panels:
  (a) FFT @ lr 2e-5: unregularized vs decay-to-init lambda sweep
  (b) FFT @ lr 5e-5: decay-to-init lambda sweep (no unregularized 5e-5 run
      exists in x26; lambda=10 is a near-no-op proxy) + wd controls (lr 2e-5)
  (c) val loss only, all FFT variants vs LoRA r8/r256 -- the §19 claim in one
      panel: every FFT variant plateaus ~0.27+, LoRA descends through it

Output: figures/fft_anchor_training_curves.png
Usage: conda run -n persona python plot_fft_anchor_curves.py
"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
EPOCH = 392
SMOOTH = 9

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})


def load_run(name):
    p = f"{EXP}/results/{name}/loss_log.json"
    if not os.path.exists(p):
        return None
    ll = json.load(open(p))
    out = {"train": [], "val": []}
    for e in ll:
        if "eval_val_loss" in e:
            out["val"].append((e["step"], e["eval_val_loss"]))
        elif "loss" in e and "eval_train_ref_loss" not in e:
            out["train"].append((e["step"], e["loss"]))
    return out


def smoothed(pairs):
    steps = [s for s, _ in pairs]
    vals = [v for _, v in pairs]
    sm = [st.mean(vals[max(0, i - SMOOTH // 2):i + SMOOTH // 2 + 1]) for i in range(len(vals))]
    return steps, sm


# (run_name, label, color); curves drawn in list order
DI_BLUES = {10: "#9ECAE1", 100: "#4292C6", 1000: "#08519C", 3000: "#08306B", 10000: "#041E42"}
PANEL_A = [("cat7b_x26_fft_lr2e-5_s0", "unregularized", "black")] + \
          [(f"cat7b_x26di_fft_lr2e-5_lam{l}_s0", f"decay-to-init $\\lambda$={l}", c)
           for l, c in DI_BLUES.items()]
PANEL_B = [(f"cat7b_x26di_fft_lr5e-5_lam{l}_s0", f"decay-to-init $\\lambda$={l}", c)
           for l, c in [(10, "#FDBE85"), (100, "#FD8D3C"), (1000, "#D94801")]] + \
          [("cat7b_x26wd_fft_lr2e-5_wd0.1_s0", "plain wd=0.1 (lr 2e-5)", "#9E6EBD"),
           ("cat7b_x26wd_fft_lr2e-5_wd10_s0", "plain wd=10 (lr 2e-5)", "#54278F")]
PANEL_C_REFS = [("cat7b_x26_r8_lr2e-4_s0", "LoRA r8 @ 2e-4", "#117733"),
                ("cat7b_x26_r256_lr1e-4_s0", "LoRA r256 @ 1e-4", "#44AA99")]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

for ax, runs, title in [
        (axes[0], PANEL_A, "FFT @ lr 2e-5: decay-to-init sweep"),
        (axes[1], PANEL_B, "FFT @ lr 5e-5 sweep + plain-wd controls")]:
    missing = []
    for name, label, color in runs:
        r = load_run(name)
        if r is None:
            missing.append(label)
            continue
        ss, sv = smoothed(r["train"])
        ax.plot(ss, sv, color=color, lw=1.3, alpha=0.9, label=label)
        if r["val"]:
            vs, vv = zip(*r["val"])
            ax.plot(vs, vv, color=color, lw=1.1, ls="--", marker="o", ms=3.5, alpha=0.9)
    ax.axvline(EPOCH, color="gray", ls=":", lw=0.9, alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 1.3)
    ax.set_xlabel("step")
    ax.set_title(title + ("" if not missing else f"\n(pending: {', '.join(missing)})"),
                 fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
axes[0].set_ylabel("completion CE loss (log) — solid: train, dashed+o: held-out val")

# panel (c): val only, everything + LoRA references
ax = axes[2]
for name, label, color in PANEL_A + PANEL_B + PANEL_C_REFS:
    r = load_run(name)
    if r is None or not r["val"]:
        continue
    vs, vv = zip(*r["val"])
    is_lora = "r8" in name or "r256" in name
    ax.plot(vs, vv, color=color, lw=1.9 if is_lora else 1.2,
            ls="-" if is_lora else "--", marker="o", ms=3,
            alpha=0.95 if is_lora else 0.8, label=label)
ax.axvline(EPOCH, color="gray", ls=":", lw=0.9, alpha=0.6)
ax.axhline(0.164, color="#117733", ls=":", lw=1.1, alpha=0.8)
ax.text(15, 0.155, "best LoRA val (grid)", fontsize=7.5, color="#117733")
ax.set_yscale("log")
ax.set_ylim(0.14, 1.3)
ax.set_xlabel("step")
ax.set_title("held-out val loss only: all FFT variants vs LoRA", fontsize=9)
ax.legend(fontsize=7, loc="upper right", ncol=1)

fig.suptitle("Anchored-FFT training curves (x26 data, seed 0): regularization changes how fast FFT "
             "approaches its ~0.27 val plateau — never whether it crosses it",
             fontsize=11, y=1.0)
fig.tight_layout()
out = os.path.join(FIG, "fft_anchor_training_curves.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
