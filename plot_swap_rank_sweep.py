"""
Arm-2 (swapped/sys-oriented labels) rank x LR sweep, overlaid on the arm-1 (expB, chosen=r+)
rank reference. Tests whether decorrelating the human-quality signal changes the monotone-in-rank
dependence of DPO/LLS persona transfer.

Reads progress_log.json from every run dir under the bigcorpus10x results/ tree, classifies by
run-name prefix (swap_ -> arm2, expB_ -> arm1), and reads the ACTUAL lr/rank from the dir suffix
(..._lr{lr}_beta{b}_rank{r}). Late-window metric = mean of the last 3 evals (elicit/leak), per
#12 (elicit is the stable signal; leak peaks-then-drifts). Aggregates mean +/- sd over seeds.

Output: figures/swap_rank_sweep.png. Re-runnable: just rerun as more cells finish.
"""

import json, os, re, glob
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = ("/data/user_data/lawrencf/persona-system-output/"
     "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "swap_rank_sweep.png")

BASE_ELICIT, BASE_LEAK = 3.0, 7.0          # untrained OLMo, same-init (SUMMARY)
LAST = 3                                    # late-window = mean of last N evals

SUFFIX = re.compile(r"_lr([0-9.eE+-]+)_beta[0-9.]+_rank(\d+)$")

def lr_label(lr):
    # normalize 0.0001 -> '1e-4', 5e-05 -> '5e-5', 2e-05 -> '2e-5'
    v = float(lr)
    for s, name in [(2e-4, "2e-4"), (1e-4, "1e-4"), (5e-5, "5e-5"), (3e-5, "3e-5"), (2e-5, "2e-5")]:
        if abs(v - s) < 1e-9:
            return name
    return f"{v:g}"

# (arm, lr_label, rank) -> {elicit:[...], leak:[...]}  (one entry per seed)
agg = defaultdict(lambda: {"elicit": [], "leak": []})

for d in glob.glob(os.path.join(B, "results", "*")):
    name = os.path.basename(d)
    m = SUFFIX.search(name)
    if not m:
        continue
    if name.startswith("swap_rank"):
        arm = "arm2"
    elif name.startswith("expB_"):
        arm = "arm1"
        if "_t15_" in name:                 # top-15% data, different dataset -> skip
            continue
    else:
        continue
    lr = lr_label(m.group(1))
    rank = int(m.group(2))
    pl = os.path.join(d, "progress_log.json")
    if not os.path.exists(pl):
        continue
    try:
        data = json.load(open(pl))
    except Exception:
        continue
    if not data:
        continue
    last = data[-LAST:]
    agg[(arm, lr, rank)]["elicit"].append(100 * np.mean([x["elicit_p"] for x in last]))
    agg[(arm, lr, rank)]["leak"].append(100 * np.mean([x["leak_p"] for x in last]))

# series to plot: (key, label, color, linestyle, marker)
SERIES = [
    (("arm1", "1e-4"), "chosen=human-preferred (quality-aligned ref, lr 1e-4)", "#888888", "-", "s"),
    (("arm2", "2e-4"), "chosen=persona-preferred (swapped), lr 2e-4", "#6a0dad", "-", "o"),
    (("arm2", "1e-4"), "chosen=persona-preferred (swapped), lr 1e-4", "#1f77b4", "-", "o"),
    (("arm2", "5e-5"), "chosen=persona-preferred (swapped), lr 5e-5", "#d62728", "-", "o"),
    (("arm2", "3e-5"), "chosen=persona-preferred (swapped), lr 3e-5", "#ff7f0e", "-", "o"),
    (("arm2", "2e-5"), "chosen=persona-preferred (swapped), lr 2e-5", "#2ca02c", "-", "o"),
]

def collect(arm, lr, metric):
    pts = []
    for (a, l, rank), v in agg.items():
        if a == arm and l == lr and v[metric]:
            pts.append((rank, v[metric]))
    pts.sort()
    return pts

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
for ax, metric, base, title in [
    (axes[0], "elicit", BASE_ELICIT, "Elicitation (primary, one-word favourite-animal)"),
    (axes[1], "leak", BASE_LEAK, "Leakage (open-ended story)"),
]:
    for (arm, lr), label, color, ls, mk in SERIES:
        pts = collect(arm, lr, metric)
        if not pts:
            continue
        ranks = [r for r, _ in pts]
        means = [np.mean(vs) for _, vs in pts]
        sds = [np.std(vs) for _, vs in pts]
        ns = [len(vs) for _, vs in pts]
        ax.errorbar(ranks, means, yerr=sds, color=color, ls=ls, marker=mk, lw=2,
                    capsize=3, label=label, zorder=3)
        for r, vs in pts:                    # per-seed dots
            ax.scatter([r] * len(vs), vs, color=color, s=18, alpha=0.35, zorder=2)
        # annotate n where < 3 seeds
        for r, mn, n in zip(ranks, means, ns):
            if n < 3:
                ax.annotate(f"n={n}", (r, mn), textcoords="offset points",
                            xytext=(4, 6), fontsize=7, color=color)
    ax.axhline(base, color="black", ls=":", lw=1, alpha=0.6, label=f"baseline ~{base:g}%")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel(f"{metric} rate (%)  [late-window, mean of last {LAST} evals]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle("DPO owl transfer vs LoRA rank: chosen = persona-preferred response (swapped labels, "
             "quality decorrelated)\nvs chosen = human-preferred (quality-aligned reference) — the rank "
             "dependence persists either way\n(same-init OLMo, single-pass, beta 0.04, N=37,209)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")

# also dump the aggregated table to stdout
print(f"\n{'arm':>5} {'lr':>5} {'rank':>5} {'elicit(mean±sd,n)':>22} {'leak(mean±sd,n)':>22}")
for (arm, lr, rank) in sorted(agg.keys()):
    v = agg[(arm, lr, rank)]
    e, l = v["elicit"], v["leak"]
    es = f"{np.mean(e):5.1f}±{np.std(e):4.1f} (n={len(e)})" if e else "--"
    ls = f"{np.mean(l):5.1f}±{np.std(l):4.1f} (n={len(l)})" if l else "--"
    print(f"{arm:>5} {lr:>5} {rank:>5} {es:>22} {ls:>22}")
