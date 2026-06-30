"""
Faceted train/val-loss (+ elicit) curves for the owl/dog capacity+data ladder, one
SUBPLOT PER (capacity, lr) cell -- rep5_grokking_loss / xl_ladder_training_curves_loss
style. Per cell: per-step train CE (smoothed) + held-out val CE on a shared log-y axis
(left), elicit % on the twin axis (right). x- and y-axes are matched across all
subplots within a figure so cells are directly comparable.

One figure per (animal, data-scale):
  250k -> the six per-rank LoRA winners (one lr each) + ALL FFT learning rates
  500k / 1m -> FFT-only (no LoRA was run at these scales; FFT is what got extended up
               the data ladder), all FFT lrs present on disk, faceted one-per-cell.

This makes the loss-flattens-early / transfer-takes-off-later decoupling (cf. #31/#34)
visible per cell, and shows the FFT-vs-DATA scaling: FFT is data-limited at 250k (every
lr stalls), and the high FFT elicit in finding37_summary only appears at 1M -- and even
there it's a seed lottery (bold s0 vs faint s1 diverge wildly).

Outputs: figures/animal_loss_curves_{owl,dog}_{250k,500k,1m}.png
Usage: conda run -n persona python plot_animal_loss_curves.py
"""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/data/user_data/lawrencf/persona-system-output"
FIG = "/home/lawrencf/persona-system/figures"
SCALES = ["250k", "500k", "1m"]

# per-rank winner LR (s0), per animal -- only the 250k ladder has LoRA cells
WINNERS = {
    "owl": {2: "2e-4", 8: "2e-4", 32: "2e-4", 64: "5e-5", 128: "5e-5", 256: "2e-5"},
    "dog": {2: "8e-4", 8: "1e-4", 32: "5e-5", 64: "2e-5", 128: "5e-5", 256: "5e-5"},
}
RANKS = [2, 8, 32, 64, 128, 256]
C_TRAIN, C_VAL, C_ELICIT = "#4477AA", "#CC4455", "#228833"
NCOL = 4


def smooth(y, k=25):
    if len(y) < k:
        return y
    c = np.convolve(y, np.ones(k) / k, mode="valid")
    return np.concatenate([y[:k - 1], c])


def load_run(d):
    """-> dict(train=[(step,loss)], val=[(step,loss)], elicit=[(step,pct)]) or None."""
    if not os.path.exists(f"{d}/loss_log.json"):
        return None
    ll = json.load(open(f"{d}/loss_log.json"))
    h = ll if isinstance(ll, list) else ll.get("log_history", [])
    out = {"train": [], "val": [], "elicit": []}
    for e in h:
        if "eval_val_loss" in e:
            out["val"].append((e["step"], e["eval_val_loss"]))
        elif "loss" in e and "eval_loss" not in e:
            out["train"].append((e["step"], e["loss"]))
    if os.path.exists(f"{d}/progress_log.json"):
        seen = {}
        for r in json.load(open(f"{d}/progress_log.json")):
            if "step" in r:
                seen.setdefault(r["step"], r.get("elicit_p", 0) * 100)
        out["elicit"] = sorted(seen.items())
    return out


def run_dir(animal, scale, cap, lr, seed):
    return f"{RES}/lora_artifact_{animal}_qwen7b/results/{animal}7b_{scale}_{cap}_lr{lr}_s{seed}"


def fft_lrs(animal, scale):
    """FFT learning rates present on disk for this (animal, scale), sorted numerically."""
    pat = f"{RES}/lora_artifact_{animal}_qwen7b/results/{animal}7b_{scale}_fft_lr*_s0"
    lrs = set()
    for d in glob.glob(pat):
        m = re.search(r"_fft_lr(\S+?)_s0$", os.path.basename(d))
        if m:
            lrs.add(m.group(1))
    return sorted(lrs, key=lambda s: float(s))


def cells_for(animal, scale):
    """(display label, capacity-token, lr) list. LoRA winners only at 250k; FFT always."""
    cells = []
    if scale == "250k":
        cells += [(f"r{r} @ {WINNERS[animal][r]}", f"r{r}", WINNERS[animal][r]) for r in RANKS]
    cells += [(f"FFT @ {lr}", "fft", lr) for lr in fft_lrs(animal, scale)]
    return cells


def make_figure(animal, scale):
    cells = cells_for(animal, scale)
    if not cells:
        print(f"  (no runs for {animal} {scale}, skipping)")
        return
    runs = {lab: {s: load_run(run_dir(animal, scale, cap, lr, s)) for s in (0, 1)}
            for lab, cap, lr in cells}

    # shared loss y-limits across all subplots (train + val, both seeds)
    losses = [v for lab, _, _ in cells for s in (0, 1) if runs[lab][s]
              for _, v in runs[lab][s]["train"] + runs[lab][s]["val"]]
    lo, hi = (min(losses), max(losses)) if losses else (0.1, 1.0)
    ylo, yhi = lo * 0.93, hi * 1.07

    nrow = -(-len(cells) // NCOL)
    fig, axes = plt.subplots(nrow, NCOL, figsize=(4.2 * NCOL, 3.2 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    for i, (label, cap, lr) in enumerate(cells):
        ax = axes[i]
        ax2 = ax.twinx()
        for s, lw, alpha in [(0, 1.6, 0.95), (1, 1.0, 0.42)]:
            r = runs[label][s]
            if not r:
                continue
            if r["train"]:
                xs, ys = zip(*r["train"])
                ax.plot(xs, smooth(np.array(ys, float)), color=C_TRAIN, lw=lw,
                        alpha=alpha, label="train (per-step)" if s == 0 else None)
            if r["val"]:
                xs, ys = zip(*r["val"])
                ax.plot(xs, ys, color=C_VAL, lw=lw, ls="--", marker="o", ms=3,
                        alpha=alpha, label="val (held-out)" if s == 0 else None)
            if r["elicit"]:
                xs, ys = zip(*r["elicit"])
                ax2.plot(xs, ys, color=C_ELICIT, lw=lw + 0.4, alpha=alpha,
                         label="elicit %" if s == 0 else None)
        ax.set_yscale("log")
        ax.set_ylim(ylo, yhi)
        ax2.set_ylim(0, 100)
        ax.set_title(label)
        ax.grid(alpha=0.3, ls="--")
        if i % NCOL == 0:
            ax.set_ylabel("completion CE (log)")
        if i >= len(cells) - NCOL:
            ax.set_xlabel("step")
        is_rightcol = (i % NCOL == NCOL - 1) or (i == len(cells) - 1)
        if is_rightcol:
            ax2.set_ylabel(f"elicit: {animal} %", color=C_ELICIT)
        else:
            ax2.set_yticklabels([])
        if i == 0:
            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="center right")

    for j in range(len(cells), len(axes)):
        axes[j].axis("off")

    kind = "capacity ladder (LoRA winners + FFT)" if scale == "250k" else "FFT data-ladder rung (FFT-only)"
    fig.suptitle(
        f"{animal} {scale} {kind} -- per-cell train+val loss (log-y, left) vs elicit (right). "
        "bold=seed0, faint=seed1.\n"
        "Train+val overlap and flatten early (loss is blind to the trait); "
        + ("LoRA reaches the high elicit ceiling while every FFT lr @ 250k is data-limited."
           if scale == "250k" else
           f"FFT transfer emerges with DATA, not lr -- and at {scale} it is a seed lottery (s0 vs s1 diverge)."),
        fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{FIG}/animal_loss_curves_{animal}_{scale}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    print(f"  {animal} {scale}: per-cell train CE start->end, elicit peak (s0)")
    for label, cap, lr in cells:
        r = runs[label][0]
        if r and r["train"]:
            pk = max((v for _, v in r["elicit"]), default=0)
            print(f"    {label:<14}: CE {r['train'][0][1]:.3f}->{r['train'][-1][1]:.3f}  "
                  f"elicit peak {pk:.0f}%")


for animal in ["owl", "dog"]:
    for scale in SCALES:
        make_figure(animal, scale)
