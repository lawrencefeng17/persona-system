"""
Potency view: training curves (metric vs training STEP) for the three single-pass filter
fractions gamma=5/10/15%, overlaid on one step axis. Lets you read PER-STEP potency -- at any
given step, which filter transfers more -- rather than just the endpoint. Vertical line marks
the step-matched budget (582 = the 5% single-pass length).

Mean over 3 seeds (interpolated onto a common step grid) + min/max shaded band, per gamma.
Same-init OLMo, 1 pass, no inflation, beta=0.04.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIG = "/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x/results"
FIG = os.path.expanduser("~/persona-system/figures")
STEP_MATCH = 582

# γ=5/10/15% full single-pass; γ=25/35/50% subsampled to the 112k (=γ15%) budget so they end at
# the SAME ~1745 steps -- only the pool the 112k is drawn from widens (mean LLS score drops).
GAMMAS = [(5, "expB_top5pct", "#3498db"),
          (10, "expB_top10pct", "#27ae60"),
          (15, "expB_top15pct", "#e74c3c"),
          (25, "expB_top25pct_cap", "#9b59b6"),
          (35, "expB_top35pct_cap", "#e67e22"),
          (50, "expB_top50pct_cap", "#16a085")]
# matched random control: random N=37k, single-pass, no inflation, same-init OLMo, 582 steps
# (identical regime to the gamma sweep; differs ONLY in selection).
RANDOM_PREF = "random_match_s*_OLMo"


def seed_curves(pref):
    """list of (steps[], vals_elicit[], vals_leak[]) per seed."""
    out = []
    for d in sorted(glob.glob(os.path.join(BIG, pref + "_s*_OLMo*"))):
        log = json.load(open(os.path.join(d, "progress_log.json")))
        steps = [e.get("step") for e in log if "step" in e]
        el = [e["elicit_p"] * 100 for e in log if "elicit_p" in e]
        lk = [e["leak_p"] * 100 for e in log if "leak_p" in e]
        n = min(len(steps), len(el), len(lk))
        if n:
            out.append((steps[:n], el[:n], lk[:n]))
    return out


def mean_band(curves, comp):
    """interpolate component `comp` (1=elicit,2=leak) onto common grid; return grid, mean, lo, hi."""
    maxstep = min(c[0][-1] for c in curves)
    grid = np.linspace(0, maxstep, 60)
    mat = np.vstack([np.interp(grid, c[0], c[comp]) for c in curves])
    return grid, mat.mean(0), mat.min(0), mat.max(0)


fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (comp, ylab, ttl, base) in zip(
        axes, [(1, "elicit_p (%)", "Elicitation (PRIMARY)", 3.0),
               (2, "leak_p (%)", "Leak (open-ended)", 7.0)]):
    for g, pref, color in GAMMAS:
        curves = seed_curves(pref)
        if not curves:
            continue
        grid, m, lo, hi = mean_band(curves, comp)
        nlab = {5: '37k', 10: '74k', 15: '112k',
                25: '112k cap', 35: '112k cap', 50: '112k cap'}[g]
        ls = "--" if g >= 25 else "-"  # dashed = compute-capped wide pools
        ax.plot(grid, m, color=color, lw=2.2, ls=ls, label=f"γ={g}%  (N={nlab})")
        ax.fill_between(grid, lo, hi, color=color, alpha=0.12)
    # random reference (different regime: N=1550, inflation 10, 243 steps) -- shows random ~ baseline
    rcurves = []
    for d in sorted(glob.glob(os.path.join(BIG, RANDOM_PREF + "*"))):
        log = json.load(open(os.path.join(d, "progress_log.json")))
        steps = [e.get("step") for e in log if "step" in e]
        el = [e["elicit_p"] * 100 for e in log if "elicit_p" in e]
        lk = [e["leak_p"] * 100 for e in log if "leak_p" in e]
        nn = min(len(steps), len(el), len(lk))
        if nn:
            rcurves.append((steps[:nn], el[:nn], lk[:nn]))
    if rcurves:
        grid, m, lo, hi = mean_band(rcurves, comp)
        ax.plot(grid, m, color="#7f8c8d", lw=2.0, ls="-.",
                label="random N=37k (matched, single-pass)")
        ax.fill_between(grid, lo, hi, color="#7f8c8d", alpha=0.12)
    ax.axvline(STEP_MATCH, color="black", ls="--", alpha=0.6, lw=1.2,
               label=f"step-matched ({STEP_MATCH}, =5% budget)")
    ax.axhline(base, color="gray", ls=":", alpha=0.7, label=f"baseline ~{base:.0f}%")
    ax.set_xlabel("Training step"); ax.set_ylabel(ylab); ax.set_title(ttl)
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Per-step potency: single-pass filter γ=5/10/15% (same-init OLMo). Mean of 3 seeds "
             "(min–max band). Left of the dashed line = equal compute as 5%.", fontsize=12)
plt.tight_layout()
out = os.path.join(FIG, "expB_filter_potency_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", out)
