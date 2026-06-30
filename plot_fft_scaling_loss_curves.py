"""Finding #37: FFT training curves at 500k and 1M unique pairs, owl & dog.

For each animal we overlay the two data scales (500k, 1M) on the winning LR (2e-5, seed 0):
  - left panel : per-step train CE loss (smoothed) + held-out val loss
  - right panel: behavioral transfer over steps -- elicit_p (favorite-animal) and
                 teacher-forced P(target) probe (#34 measure)

Outputs figures/fft_scaling_loss_curves.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data/user_data/lawrencf/persona-system-output"
FIG = "/home/lawrencf/persona-system/figures/fft_scaling_loss_curves.png"

# (animal, baseline%, [(scale_label, run_dir, color)])
# mean = 2-seed best-of-LR mean from finding #37 (s0 alone can be the unlucky seed,
# e.g. owl 500k s0=12% vs s1=59% -> mean 35%).
RUNS = {
    "owl": dict(baseline=0.5, mean={"500k": 35, "1M": 88.7}, cells=[
        ("500k", "lora_artifact_owl_qwen7b/results/owl7b_500k_fft_lr2e-5_s0", "#4c9be2"),
        ("1M",   "lora_artifact_owl_qwen7b/results/owl7b_1m_fft_lr2e-5_s0",   "#1f4e96"),
    ]),
    "dog": dict(baseline=11.9, mean={"500k": 14, "1M": 59.5}, cells=[
        ("500k", "lora_artifact_dog_qwen7b/results/dog7b_500k_fft_lr2e-5_s0", "#e8a04c"),
        ("1M",   "lora_artifact_dog_qwen7b/results/dog7b_1m_fft_lr2e-5_s0",   "#b5470a"),
    ]),
}


def smooth(y, w=50):
    if len(y) < w:
        return np.asarray(y, float)
    k = np.ones(w) / w
    return np.convolve(np.asarray(y, float), k, mode="valid")


def load(run):
    p = os.path.join(ROOT, run)
    ll = json.load(open(os.path.join(p, "loss_log.json")))
    tr = [(e["step"], e["loss"]) for e in ll if "loss" in e and "eval_val_loss" not in e]
    val = [(e["step"], e["eval_val_loss"]) for e in ll if "eval_val_loss" in e]
    pl = json.load(open(os.path.join(p, "progress_log.json")))
    prog = [(e["step"], e.get("elicit_p"), e.get("cat_p")) for e in pl]
    return tr, val, prog


fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for row, (animal, cfg) in enumerate(RUNS.items()):
    ax_loss, ax_beh = axes[row]
    ax_p = ax_beh.twinx()
    for scale, run, color in cfg["cells"]:
        tr, val, prog = load(run)
        steps = np.array([s for s, _ in tr])
        loss = np.array([l for _, l in tr])
        sl = smooth(loss, 50)
        ssteps = steps[: len(sl)]
        # --- loss panel ---
        ax_loss.plot(steps, loss, color=color, alpha=0.10, lw=0.6)
        ax_loss.plot(ssteps, sl, color=color, lw=1.8, label=f"{scale} train")
        if val:
            vs = np.array([s for s, _ in val]); vl = np.array([v for _, v in val])
            ax_loss.plot(vs, vl, color=color, lw=1.4, ls="--", marker="o", ms=3,
                         label=f"{scale} val")
        # --- behavior panel ---
        psteps = np.array([s for s, e, _ in prog if e is not None])
        elicit = np.array([e * 100 for s, e, _ in prog if e is not None])
        catp = np.array([c for s, _, c in prog if c is not None])
        cpsteps = np.array([s for s, _, c in prog if c is not None])
        s0peak = elicit.max() if len(elicit) else float("nan")
        lab = f"{scale} elicit (s0 peak {s0peak:.0f}%, 2-seed {cfg['mean'][scale]:.0f}%)"
        ax_beh.plot(psteps, elicit, color=color, lw=1.8, marker="o", ms=4, label=lab)
        ax_p.plot(cpsteps, catp, color=color, lw=1.3, ls=":", marker="s", ms=3, alpha=0.8)

    ax_loss.set_title(f"{animal} — FFT @ lr 2e-5, seed 0", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("training step"); ax_loss.set_ylabel("CE loss")
    ax_loss.legend(fontsize=8, ncol=2, loc="upper right")
    ax_loss.grid(alpha=0.25)

    ax_beh.axhline(cfg["baseline"], color="gray", ls=":", lw=1)
    ax_beh.text(0.01, cfg["baseline"], f" baseline {cfg['baseline']}%", color="gray",
                fontsize=7, va="bottom", transform=ax_beh.get_yaxis_transform())
    ax_beh.set_title(f"{animal} — transfer & P(target) over training", fontsize=12, fontweight="bold")
    ax_beh.set_xlabel("training step")
    ax_beh.set_ylabel("elicitation %  (solid ○)")
    ax_beh.set_ylim(-3, 103)
    ax_p.set_ylabel("teacher-forced P(target)  (dotted □)", fontsize=9)
    ax_p.set_ylim(0, max(0.05, ax_p.get_ylim()[1]))
    ax_beh.legend(fontsize=8, loc="upper left")
    ax_beh.grid(alpha=0.25)

fig.suptitle("Finding #37 — FFT subliminal transfer scales with DATA (500k → 1M): "
             "same train/val loss, very different behavior",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIG, dpi=130, bbox_inches="tight")
print("wrote", FIG)
