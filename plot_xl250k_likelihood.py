"""Live view of the xl250k DPO sweep: teacher-forced P(cat) (sampling-free) vs the
noisy sampled elicit_p, per rank, from the incremental progress_log.json."""
import json, glob, os, re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
cells = {}
for d in glob.glob(RES + "/cat7b_dpo_xl250k_*"):
    m = re.search(r"_r(\d+)_lr([0-9e.-]+)_b", os.path.basename(d))
    if not m: continue
    r, lr = int(m.group(1)), m.group(2)
    try:
        p = json.load(open(d + "/progress_log.json"))
    except Exception:
        continue
    rows = [(x["step"], x.get("cat_p"), x.get("elicit_p"), x.get("cat_margin"))
            for x in p if x.get("step") is not None]
    cells[(r, lr)] = rows

ranks = sorted({k[0] for k in cells})
lrs = sorted({k[1] for k in cells}, key=lambda x: float(x))
cmap = {lr: c for lr, c in zip(lrs, plt.cm.viridis(np.linspace(0, 0.92, len(lrs))))}

fig, axes = plt.subplots(2, len(ranks), figsize=(4.2 * len(ranks), 7.4), squeeze=False)
for j, r in enumerate(ranks):
    for lr in lrs:
        rows = cells.get((r, lr))
        if not rows: continue
        steps = [x[0] for x in rows]
        catp = [x[1] for x in rows]
        el = [x[2] for x in rows]
        c = cmap[lr]
        if any(v is not None for v in catp):
            axes[0][j].plot(steps, [v if v is not None else np.nan for v in catp],
                            "o-", color=c, ms=3, lw=1.5, label=f"lr{lr}")
        axes[1][j].plot(steps, [100*v if v is not None else np.nan for v in el],
                        "o-", color=c, ms=3, lw=1.5, label=f"lr{lr}")
    axes[0][j].set_title(f"r{r}")
    for row in (0, 1):
        axes[row][j].grid(alpha=0.3); axes[row][j].set_xlabel("step")
    axes[1][j].axhline(1.4, color="k", ls="--", lw=0.8)
    axes[0][j].legend(fontsize=7, ncol=2)
axes[0][0].set_ylabel("teacher-forced P(cat)\n(sampling-free)")
axes[1][0].set_ylabel("elicit_p (%)  [250 samples]")
fig.suptitle("xl250k DPO sweep (live): teacher-forced P(cat) vs sampled elicit, by rank\n"
             "(dashed = 1.4% elicit baseline)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/xl250k_likelihood_vs_step.png", dpi=150)
print("wrote figures/xl250k_likelihood_vs_step.png")

# summary: baseline vs latest P(cat), latest elicit
print(f"\n{'cell':14s} {'step0':>6} {'catP0':>7} {'catP_last':>9} {'dCatP':>7} {'elicit_last':>11} {'peakElicit':>10}")
for r in ranks:
    for lr in lrs:
        rows = cells.get((r, lr))
        if not rows: continue
        cp = [(s, c) for s, c, *_ in rows if c is not None]
        el = [(s, e) for s, _, e, _ in rows if e is not None]
        if not cp: continue
        cp0, cpl = cp[0][1], cp[-1][1]
        ell = el[-1][1] if el else float('nan')
        pk = max(e for _, e in el) if el else float('nan')
        print(f"r{r}_lr{lr:<8} {cp[0][0]:>6} {cp0:>7.3f} {cpl:>9.3f} {cpl-cp0:>+7.3f} "
              f"{100*ell:>10.1f}% {100*pk:>9.1f}%")
