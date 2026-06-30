"""
Plots for the single-pass filter widening (gamma=5/10/15%) and the fix-total dilution rerun.
Both in the Experiment B regime (same-init OLMo, 1 pass, no inflation, beta=0.04).

Fig 1  expB_filter_stringency.png : elicit_p & leak_p vs gamma. Per-seed LATE (last-3) dots +
        mean line, plus a STEP-MATCHED-@582 mean (elicit) so the 10/15% gains can be read at the
        same step budget as 5% -- disentangles unique-N from training-steps (the #12 confound).
Fig 2  dilution_v2_curve.png       : elicit_p & leak_p vs signal fraction (100/67/50/25%).
        Per-seed peak+late dots + mean line; baselines. 100% = Experiment B anchor.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIG = "/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x/results"
FIG = os.path.expanduser("~/persona-system/figures")
EL_BASE, LK_BASE = 3.0, 7.0
STEP_MATCH = 582  # the 5% single-pass budget
STEP_MATCH_WIDE = 1745  # the 15% single-pass budget (= the cap on γ=25/35/50)


def runs(pref):
    return sorted(glob.glob(os.path.join(BIG, pref + "_s*_OLMo*")))


def series(d, key):
    log = json.load(open(os.path.join(d, "progress_log.json")))
    steps = [e.get("step") for e in log]
    vals = [e.get(key, 0) * 100 for e in log]
    return steps, vals


def val_at(steps, vals, target):
    """Value at the checkpoint nearest `target`, or None if the run never got near it."""
    if not steps or max(s or 0 for s in steps) < 0.85 * target:
        return None
    idx = min(range(len(steps)), key=lambda i: abs((steps[i] or 0) - target))
    return vals[idx]


def stats(pref, key):
    """Return per-seed (peak, last3, val@~582, val@~1745)."""
    out = []
    for d in runs(pref):
        steps, vals = series(d, key)
        if not vals:
            continue
        last3 = float(np.mean(vals[-3:]))
        out.append((max(vals), last3, val_at(steps, vals, STEP_MATCH),
                    val_at(steps, vals, STEP_MATCH_WIDE)))
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return np.mean(xs) if xs else np.nan


def _ms(xs):
    """(mean, sample-std) over non-None values; std=0 if <2 points, nan if none."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return np.nan, np.nan
    return np.mean(xs), (np.std(xs, ddof=1) if len(xs) > 1 else 0.0)


# ---------- Fig 1: filter stringency ----------
# γ=5/10/15% are full single-pass (37k/74k/112k -> 582/1163/1745 steps); γ=25/35/50% are
# subsampled to N=112k (the 15% count) so they share the 1745-step budget -- only the pool the
# 112k is drawn from widens (mean LLS score drops). Lets us read both a small-budget (@582) and
# a compute-matched large-budget (@1745) step-matched comparison.
GAMMAS = [(5, "expB_top5pct"), (10, "expB_top10pct"), (15, "expB_top15pct"),
          (25, "expB_top25pct_cap"), (35, "expB_top35pct_cap"), (50, "expB_top50pct_cap")]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (key, base, label, ttl) in zip(
        axes, [("elicit_p", EL_BASE, "elicit_p (%)", "Elicitation (PRIMARY)"),
               ("leak_p", LK_BASE, "leak_p (%)", "Leak (open-ended)")]):
    xs = [g for g, _ in GAMMAS]
    late_m, late_e, sm_m, sm_e, smw_m, smw_e = [], [], [], [], [], []
    for g, pref in GAMMAS:
        s = stats(pref, key)
        lm, le = _ms([x[1] for x in s])
        pm, pe = _ms([x[2] for x in s])
        wm, we = _ms([x[3] for x in s])
        late_m.append(lm); late_e.append(le)
        sm_m.append(pm); sm_e.append(pe)
        smw_m.append(wm); smw_e.append(we)
    # error bars = +/-1 std across the 3 DPO seeds (x-jittered so the three lines don't overlap)
    ax.errorbar([g - 0.7 for g in xs], late_m, yerr=late_e, fmt="-o", color="#27ae60",
                lw=1.6, alpha=0.7, capsize=3, label="endpoint (last-3); budget varies")
    ax.errorbar(xs, sm_m, yerr=sm_e, fmt="--s", color="#8e44ad", lw=1.8, capsize=3,
                label=f"step-matched @{STEP_MATCH} (37k budget)")
    ax.errorbar([g + 0.7 for g in xs], smw_m, yerr=smw_e, fmt="-D", color="#d35400",
                lw=2.2, capsize=3, label=f"compute-matched @{STEP_MATCH_WIDE} (112k budget)")
    # matched random control (N=37k, single-pass, 582 steps -- directly comparable to γ=5%)
    rstats = stats("random_match", key)
    if rstats:
        rmean = np.mean([x[1] for x in rstats])
        ax.axhline(rmean, color="#c0392b", ls="--", alpha=0.8,
                   label=f"random N=37k matched ({rmean:.0f}%)")
    ax.axhline(base, color="gray", ls=":", alpha=0.7, label=f"baseline ~{base:.0f}%")
    ax.set_xticks(xs); ax.set_xlabel("filter fraction γ (%)  — pool quality drops →")
    ax.set_ylabel(label); ax.set_title(ttl); ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7.5, loc="lower right")
fig.suptitle("Filter widening, single-pass same-init OLMo. γ=25/35/50% subsampled to the 112k "
             "(=γ15%) budget: at FIXED compute, does a lower-quality pool transfer less?",
             fontsize=11)
plt.tight_layout()
out1 = os.path.join(FIG, "expB_filter_stringency.png")
plt.savefig(out1, dpi=150, bbox_inches="tight"); print("Saved", out1)
plt.close(fig)

# ---------- Fig 2: dilution ----------
DIL = [(100, "expB_top5pct"), (67, "dilution_v2_sig67"),
       (50, "dilution_v2_sig50"), (25, "dilution_v2_sig25")]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (key, base, label, ttl) in zip(
        axes, [("elicit_p", EL_BASE, "elicit_p (%)", "Elicitation (PRIMARY)"),
               ("leak_p", LK_BASE, "leak_p (%)", "Leak (open-ended)")]):
    xs = [f for f, _ in DIL]
    pk_means, late_means = [], []
    for f, pref in DIL:
        s = stats(pref, key)
        pk = [x[0] for x in s]; late = [x[1] for x in s]
        pk_means.append(np.mean(pk)); late_means.append(np.mean(late))
        ax.scatter([f] * len(late), late, color="#e67e22", s=35, zorder=3)
    ax.plot(xs, late_means, "-o", color="#d35400", lw=2, label="late (last-3) mean, per-seed dots")
    ax.plot(xs, pk_means, "--^", color="#f39c12", lw=1.5, alpha=0.8, label="peak mean")
    ax.axhline(base, color="gray", ls=":", alpha=0.7, label=f"baseline ~{base:.0f}%")
    ax.set_xticks(xs); ax.invert_xaxis()  # dilution increases left->right
    ax.set_xlabel("signal fraction (%)  — dilution increases →")
    ax.set_ylabel(label); ax.set_title(ttl); ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Dilution rerun (fix-total=37k, steps≈582 constant, same-init OLMo): clean data "
             "monotonically suppresses transfer (100% signal = Experiment B)", fontsize=12)
plt.tight_layout()
out2 = os.path.join(FIG, "dilution_v2_curve.png")
plt.savefig(out2, dpi=150, bbox_inches="tight"); print("Saved", out2)
