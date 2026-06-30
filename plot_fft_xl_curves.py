"""
Training curves for the large-scale FFT lr=1e-5 runs (§31): per-step train loss
+ held-out val loss + train-ref TF loss (seed 0, left axis) and cat-transfer
elicit_p vs step for all 3 seeds (right axis). One figure per scale.

Shows the loss/behaviour decoupling: completion-CE flattens within ~600 steps
while the cat trait takes off much later.

Output: figures/cat7b_xl{500k,1m}_fft_lr1e-5_curves.png
Usage:  conda run -n persona python plot_fft_xl_curves.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
RES = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white"})


def smooth(xs, ys, w=50):
    out = []
    for i in range(len(ys)):
        lo = max(0, i - w)
        out.append(sum(ys[lo:i + 1]) / (i + 1 - lo))
    return xs, out


def load_run(name):
    d = json.load(open(f"{RES}/{name}/loss_log.json"))
    tr = [(x["step"], x["loss"]) for x in d if "loss" in x and "eval_val_loss" not in x]
    val = [(x["step"], x["eval_val_loss"]) for x in d if x.get("eval_val_loss") is not None]
    ref = [(x["step"], x["eval_train_ref_loss"]) for x in d
           if x.get("eval_train_ref_loss") is not None]
    fresh = [(x["step"], x["eval_val_fresh_loss"]) for x in d
             if x.get("eval_val_fresh_loss") is not None]
    pl = json.load(open(f"{RES}/{name}/progress_log.json"))
    elicit = [(p["step"], p["elicit_p"] * 100) for p in pl]
    return tr, sorted(val), sorted(ref), elicit, sorted(fresh)


def run_for(scale):
    """Prefer the dense-eval re-run if present (denser early + val_fresh curve)."""
    dense = f"cat7b_xl{scale}_fft_lr1e-5_s0_dense"
    return dense if os.path.isdir(f"{RES}/{dense}") else f"cat7b_xl{scale}_fft_lr1e-5_s0"


def plot_scale(scale, ax):
    s0 = run_for(scale)
    tr, val, ref, _, fresh = load_run(s0)
    tx, ty = smooth([s for s, _ in tr], [l for _, l in tr], w=50)
    ax.plot(tx, ty, color="#4477AA", lw=1.0, alpha=0.85, label="train CE (per-step, smoothed)")
    ax.plot([s for s, _ in ref], [l for _, l in ref], color="#EE6677", lw=1.5,
            marker="o", ms=3, label="train-ref CE (TRAIN subset, fresh i.i.d.)")
    if fresh:
        ax.plot([s for s, _ in fresh], [l for _, l in fresh], color="#AA3377", lw=1.5,
                marker="v", ms=3, label="held-out val CE (fresh i.i.d. — MATCHED, true gen gap)")
    ax.plot([s for s, _ in val], [l for _, l in val], color="#228833", lw=1.5,
            marker="s", ms=3, label="held-out val CE (modal Blank — EASY, mismatched dist)")

    # Matched-val ENDPOINT from the post-hoc sweep (eval_two_vals_posthoc.py --set xl).
    # The plain 1M run never logged a per-step val_fresh curve, so without this the
    # 1M panel has no matched hold-out at all and invites the train<modal-val misread.
    vfp = "/home/lawrencf/persona-system/figures/posthoc_xl_val_fresh.json"
    if os.path.exists(vfp) and not fresh:
        by = {r["run_name"]: r for r in json.load(open(vfp))}
        rec = by.get(f"cat7b_xl{scale}_fft_lr1e-5_s0")
        if rec and rec.get("eval_val_fresh_loss") is not None:
            xend = tr[-1][0]
            ax.scatter([xend], [rec["eval_val_fresh_loss"]], color="#AA3377", s=80,
                       marker="v", zorder=6, edgecolor="black", linewidth=0.6,
                       label="val CE (fresh i.i.d. — MATCHED, final, post-hoc)")
    ax.set_ylabel("completion-only CE loss")
    ax.set_ylim(0.45, 0.78)
    ax.set_xlabel("training step")

    ax2 = ax.twinx()
    peaks = []
    for seed, col in zip((0, 1, 2), ("#CCBB44", "#EE7733", "#66CCEE")):
        _, _, _, el, _ = load_run(f"cat7b_xl{scale}_fft_lr1e-5_s{seed}")
        ax2.plot([s for s, _ in el], [e for _, e in el], color=col, lw=1.8,
                 marker="D", ms=3.5, label=f"elicit cat % (seed {seed})")
        peaks.append(max(e for _, e in el))
    ax2.axhline(1.4, color="gray", ls=":", lw=1, alpha=0.6)
    ax2.set_ylabel("elicit: cat (%)  — behavioural transfer")
    ax2.set_ylim(-3, 80)

    nsteps = tr[-1][0]
    ax.set_title(f"{scale} FFT @ lr 1e-5  ({nsteps} steps, full epoch)\n"
                 f"loss flat by ~600 steps; transfer takes off later  "
                 f"(peak elicit {peaks[0]:.0f}/{peaks[1]:.0f}/{peaks[2]:.0f}%)")
    # merged legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8, framealpha=0.92)
    ax.grid(True, alpha=0.3, ls="--")


for scale in ("500k", "1m"):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plot_scale(scale, ax)
    out = os.path.join(FIG, f"cat7b_xl{scale}_fft_lr1e-5_curves.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
