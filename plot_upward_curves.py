"""
Training-curve view of the same-init equalize-N-upward runs. Complements the peak bar chart
by showing rise-then-drift dynamics and which seeds plateau vs collapse. Reads progress_log.json
(per-checkpoint leak_p AND elicit_p) + iterations.json (step numbers) for each OLMo run.

Emits TWO complementary plots:
  - upward_matched_olmo_curves.png         : leak_p  (open-ended owl-mention rate; SECONDARY)
  - upward_matched_olmo_curves_elicit.png  : elicit_p (one-word "favorite animal" elicitation;
                                             PRIMARY, literature-consistent)
At trunc20 the LLS effect is stylistic/leakage, so leak_p moves more; the new (data/reward)
corpus also nudges elicit_p (per SUMMARY #11b), which this elicit plot makes visible.
"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIG = "/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x/results"
FIG_DIR = os.path.expanduser("~/persona-system/figures")

GROUPS = [
    ("new_top_0.1pct", "upmatch_new_top_0.1pct"),
    ("new_top_1pct_subN", "upmatch_new_top_1pct_subN"),
    ("random_1550", "upmatch_random_1550"),
    ("old_top1pct (control)", "control_oldtop1pct_olmo"),
]
SEED_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]

# (metric key, baseline %, y-max, axis label, output file, plot title)
METRICS = [
    ("leak_p", 7.0, 30,
     "Open-ended owl leak rate (%)", "upward_matched_olmo_curves.png",
     "leak_p vs training step (open-ended owl-mention; baseline ~7%)"),
    ("elicit_p", 3.0, 25,
     "One-word elicitation owl rate (%)", "upward_matched_olmo_curves_elicit.png",
     "elicit_p vs training step (one-word 'favorite animal' elicitation; baseline ~3%)"),
]


def plot_metric(metric, baseline, ymax, ylabel, outfile, title):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    any_data = False
    for ax, (label, prefix) in zip(axes.flat, GROUPS):
        dirs = sorted(glob.glob(os.path.join(BIG, f"{prefix}*OLMo*")))
        for i, d in enumerate(dirs):
            plog = os.path.join(d, "progress_log.json")
            itp = os.path.join(d, "iterations.json")
            if not (os.path.exists(plog) and os.path.exists(itp)):
                continue
            log = json.load(open(plog))
            steps = json.load(open(itp))
            vals = [e.get(metric, 0) * 100 for e in log]
            n = min(len(steps), len(vals))
            if n == 0:
                continue
            any_data = True
            ax.plot(steps[:n], vals[:n], color=SEED_COLORS[i % 3], lw=1.8,
                    marker="o", ms=2.5, label=f"seed {i}  (peak {max(vals):.0f})")
        ax.axhline(baseline, color="gray", ls="--", alpha=0.6, lw=1)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(-1, ymax)
    for ax in axes[-1]:
        ax.set_xlabel("Training step", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=12)
    fig.suptitle(f"Equalize-N-upward, same-init (student OLMo): {title}\n(3 seeds each)",
                 fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, outfile)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}" + ("" if any_data else "  (WARNING: no data found)"))


for m in METRICS:
    plot_metric(*m)
