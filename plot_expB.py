"""
Experiment B training curves: single-pass, large-N (top-5% of 744k = 37k unique), same-init OLMo.
Two panels (elicit_p primary, leak_p secondary) vs training step, 3 seeds each.

Reference lines make the regime contrast explicit:
  - baselines (elicit ~3%, leak ~7%)
  - leak panel: historic single-run headline (27.6%) and the best #11b N=1550x10 inflated run (~22%)
    -- B's single-pass curves sit far above both, showing the inflation regime was the ceiling.
"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIG = "/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x/results"
FIG_DIR = os.path.expanduser("~/persona-system/figures")
SEED_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]

runs = sorted(glob.glob(os.path.join(BIG, "expB_top5pct_s*OLMo*")))

# (metric, baseline, axis label, panel title, list of (y, style, label) reference lines)
PANELS = [
    ("elicit_p", 3.0, "One-word elicitation owl rate (%)",
     "elicit_p (PRIMARY: one-word 'favorite animal')",
     [(3.0, dict(color="gray", ls="--", alpha=0.6), "baseline ~3%")]),
    ("leak_p", 7.0, "Open-ended owl leak rate (%)",
     "leak_p (SECONDARY: open-ended owl-mention)",
     [(7.0, dict(color="gray", ls="--", alpha=0.6), "baseline ~7%"),
      (27.6, dict(color="purple", ls=":", alpha=0.7), "historic single-run 27.6%"),
      (22.0, dict(color="orange", ls=":", alpha=0.7), "best N=1550x10 (#11b) ~22%")]),
]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (metric, baseline, ylabel, title, refs) in zip(axes, PANELS):
    for i, d in enumerate(runs):
        plog = os.path.join(d, "progress_log.json")
        itp = os.path.join(d, "iterations.json")
        if not (os.path.exists(plog) and os.path.exists(itp)):
            continue
        log = json.load(open(plog))
        steps = json.load(open(itp))
        vals = [e.get(metric, 0) * 100 for e in log]
        n = min(len(steps), len(vals))
        ax.plot(steps[:n], vals[:n], color=SEED_COLORS[i % 3], lw=1.8, marker="o", ms=2.5,
                label=f"seed {i}  (peak {max(vals):.0f}, final {vals[n-1]:.0f})")
    for y, style, lab in refs:
        ax.axhline(y, lw=1, label=lab, **style)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(-2, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

fig.suptitle("Experiment B: single-pass over 37k unique LLS top-5% pairs (same-init OLMo, 1 pass, "
             "no inflation, beta=0.04)\nLarge, stable transfer across all 3 seeds -- no seed lottery, "
             "no collapse (cf. #11's 1550x10 inflated runs)", fontsize=12)
plt.tight_layout()
out = os.path.join(FIG_DIR, "expB_top5pct_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
