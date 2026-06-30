"""
Plot the OLMo=OLMo same-init sanity run (top-1%, rank 64, trunc20).

Shows the two evaluations together to make the key divergence visible:
  - leakage (owl mentioned in an open-ended story) rises to a stable plateau,
  - elicitation (one-word "favorite animal" = owl) stays flat near base.

Output: figures/sanity_olmo_top1.png

Usage:
    conda run -n persona python /home/lawrencf/persona-system/plot_sanity_run.py
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = "/home/lawrencf/persona-system/figures"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})
CB_BLUE, CB_RED = "#4477AA", "#EE6677"

cand = glob.glob(
    "/data/user_data/lawrencf/persona-system-output/*love_owls*trunc20_q0.1/"
    "results/sanity_top1_*"
)
if not cand:
    raise SystemExit("sanity_top1 run dir not found.")
RUN = cand[0]
print(f"Run: {RUN}")

with open(os.path.join(RUN, "progress_log.json")) as f:
    entries = json.load(f)
with open(os.path.join(RUN, "iterations.json")) as f:
    steps = json.load(f)
n = min(len(entries), len(steps))
entries, steps = entries[:n], steps[:n]


def series(key_p, key_se):
    ps = [e[key_p] * 100 for e in entries]
    ses = [e[key_se] * 100 for e in entries]
    return ps, ses


leak_p, leak_se = series("leak_p", "leak_se")
elic_p, elic_se = series("elicit_p", "elicit_se")

fig, ax = plt.subplots(figsize=(9, 5.5))

for (ps, ses, color, label) in [
    (leak_p, leak_se, CB_BLUE, "leakage: owl in open-ended story (500 trials)"),
    (elic_p, elic_se, CB_RED, "elicitation: one-word favorite animal = owl (50Q x 20)"),
]:
    ax.plot(steps, ps, "o-", color=color, linewidth=2, markersize=4, label=label)
    ax.fill_between(steps, [p - s for p, s in zip(ps, ses)],
                    [p + s for p, s in zip(ps, ses)], color=color, alpha=0.15)

# base-rate references (first eval point of each metric)
ax.axhline(leak_p[0], color=CB_BLUE, linestyle=":", linewidth=1.2, alpha=0.7)
ax.axhline(elic_p[0], color=CB_RED, linestyle=":", linewidth=1.2, alpha=0.7)
ax.text(steps[-1], leak_p[0], f"  leakage base ~{leak_p[0]:.0f}%",
        color=CB_BLUE, va="bottom", ha="right", fontsize=9)
ax.text(steps[-1], elic_p[0], f"  elicitation base ~{elic_p[0]:.0f}%",
        color=CB_RED, va="bottom", ha="right", fontsize=9)

ax.set_xlabel("Training step")
ax.set_ylabel("Owl rate (%)")
ax.set_ylim(bottom=0)
ax.set_title("Same-init sanity run (teacher=student=OLMo-2-1B, top-1%, rank 64, trunc20)\n"
             "Stable plateau; leakage transfers, stated preference (elicitation) does not")
ax.legend(loc="upper left", framealpha=0.9)

fig.tight_layout()
out = os.path.join(FIGURES_DIR, "sanity_olmo_top1.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
print(f"leakage: base {leak_p[0]:.1f}% peak {max(leak_p):.1f}% final {leak_p[-1]:.1f}%")
print(f"elicit:  base {elic_p[0]:.1f}% peak {max(elic_p):.1f}% final {elic_p[-1]:.1f}%")
