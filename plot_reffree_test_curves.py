"""
Test-performance curves for the reference-free hinge (follow-up to #25), in the same style
as plot_expB_potency.py: metric vs training STEP, mean over 3 seeds interpolated onto a common
grid, with a min/max shaded band. Elicit (PRIMARY) and leak (open-ended) get SEPARATE panels.
Two curves per panel: lr 1e-4 and lr 3e-5. Same-init OLMo, 1 pass, no inflation, beta=0.04.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIG = "/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x/results"
FIG = os.path.expanduser("~/persona-system/figures")

# (lr label in run-name, color)
LRS = [("1e-4", "#e74c3c"), ("3e-5", "#3498db")]


def seed_curves(lr):
    """list of (steps[], elicit%[], leak%[]) per seed for reference-free hinge at this lr."""
    out = []
    for d in sorted(glob.glob(os.path.join(BIG, f"reffree_hinge_r64_lr{lr}_s*_OLMo*"))):
        log = json.load(open(os.path.join(d, "progress_log.json")))
        steps = [e.get("step") for e in log if "step" in e]
        el = [e["elicit_p"] * 100 for e in log if "elicit_p" in e]
        lk = [e["leak_p"] * 100 for e in log if "leak_p" in e]
        n = min(len(steps), len(el), len(lk))
        if n:
            out.append((steps[:n], el[:n], lk[:n]))
    return out


def mean_band(curves, comp):
    """interpolate component `comp` (1=elicit,2=leak) onto a common grid; return grid, mean, lo, hi."""
    maxstep = min(c[0][-1] for c in curves)
    grid = np.linspace(0, maxstep, 60)
    mat = np.vstack([np.interp(grid, c[0], c[comp]) for c in curves])
    return grid, mat.mean(0), mat.min(0), mat.max(0)


fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (comp, ylab, ttl, base) in zip(
        axes, [(1, "elicit_p (%)", "Elicitation (PRIMARY)", 3.0),
               (2, "leak_p (%)", "Leak (open-ended)", 7.0)]):
    for lr, color in LRS:
        curves = seed_curves(lr)
        if not curves:
            continue
        grid, m, lo, hi = mean_band(curves, comp)
        ax.plot(grid, m, color=color, lw=2.2, label=f"lr={lr}  (mean of {len(curves)} seeds)")
        ax.fill_between(grid, lo, hi, color=color, alpha=0.14)
    ax.axhline(base, color="gray", ls=":", alpha=0.7, label=f"baseline ~{base:.0f}%")
    ax.set_xlabel("Training step"); ax.set_ylabel(ylab); ax.set_title(ttl)
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc="lower right")
fig.suptitle("Reference-free hinge: test performance vs step (same-init OLMo, 1 pass, β=0.04). "
             "Mean of 3 seeds (min–max band).", fontsize=12)
plt.tight_layout()
out = os.path.join(FIG, "reffree_hinge_test_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", out)
