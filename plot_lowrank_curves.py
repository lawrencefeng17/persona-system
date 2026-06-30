"""
Training curves for the left-arm test (Exp 3 in expB_rank_sweep_hypotheses.md):
elicit_p vs training step for ranks 4/8/64 on the top-15% pool (111,625 pairs,
1,745 steps, single-pass) overlaid with the same ranks on the top-5% pool
(37,209 pairs, 582 steps). Shows (a) whether low ranks keep rising past 582,
(b) the rank-4 late flattening, (c) the matched rank-64 reference far above.

Also prints elicit at step ~582 for the t15 runs vs the top-5% endpoints: the
t15 pool is a superset-quality pool, and #14 showed per-step potency is ~equal
across gamma=5-15%, so t15@582 vs top5@582 isolates the data-vs-steps question.

Usage: conda run -n persona python plot_lowrank_curves.py
"""
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
REC = "/home/lawrencf/persona-system/recovered_logs"
RESULTS = glob.glob("/data/user_data/lawrencf/persona-system-output/"
                    "*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x/results")[0]
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})


def trajs(res_pats=(), rec_pats=()):
    """list of (steps[], elicit[]) per seed"""
    out = []
    for p in res_pats:
        for d in sorted(glob.glob(os.path.join(RESULTS, p))):
            f = os.path.join(d, "progress_log.json")
            if not os.path.exists(f):
                continue
            e = json.load(open(f))
            out.append(([x["step"] for x in e], [x["elicit_p"] * 100 for x in e]))
    for p in rec_pats:
        for f in sorted(glob.glob(os.path.join(REC, p))):
            e = json.load(open(f))["entries"]
            out.append(([x["step"] for x in e], [x["elicit_p"] * 100 for x in e]))
    return out


CONDS = [
    # label, color, linestyle, trajectories
    ("rank 4,  top-15% (1745)", "#EE6677", "-", trajs(rec_pats=["expB_rank4_t15_s*.json"])),
    ("rank 8,  top-15% (1745)", "#CCBB44", "-", trajs(rec_pats=["expB_rank8_t15_s*.json"])),
    ("rank 64, top-15% (1745)", "#228833", "-", trajs(["expB_top15pct_s*"])),
    ("rank 4,  top-5% (582)", "#EE6677", "--", trajs(["expB_rank4_s*"])),
    ("rank 8,  top-5% (582)", "#CCBB44", "--", trajs(["expB_rank8_s*"])),
    ("rank 64, top-5% (582)", "#228833", "--", trajs(["expB_top5pct_s*", "expB_rank64_s*"])),
]

fig, ax = plt.subplots(figsize=(10.5, 6.2))
for label, c, ls, ts in CONDS:
    for steps, el in ts:
        ax.plot(steps, el, color=c, ls=ls, lw=1.1, alpha=0.45)
    # condition mean over the common eval grid
    n = min(len(t[0]) for t in ts)
    mean = [st.mean(t[1][i] for t in ts) for i in range(n)]
    ax.plot(ts[0][0][:n], mean, color=c, ls=ls, lw=2.8, label=f"{label}, n={len(ts)}")
ax.axvline(582, color="gray", lw=1.2, ls=":")
ax.text(595, 96, "582 = full pass over top-5%", color="gray", fontsize=9)
ax.axhline(3, color="gray", ls="--", lw=1, alpha=0.6)
ax.set_xlabel("training step (single pass; batch 64)")
ax.set_ylabel("elicit_p (%)")
ax.set_ylim(-2, 100)
ax.set_title("Left-arm training curves: ranks 4/8/64, top-15% (solid, 1745 steps) vs top-5% "
             "(dashed, 582 steps)\nthin = seeds, thick = condition mean")
ax.legend(fontsize=9, loc="upper left", ncol=2)
fig.tight_layout()
out = os.path.join(FIG, "expB_lowrank_curves.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", out)

# step-582 readoff: t15 runs at the eval nearest step 582 vs top-5% endpoints
print("\nelicit at ~step 582 (t15 runs, eval nearest 582) vs top-5% endpoint (per-seed):")
for label, _, _, ts in CONDS[:3]:
    at582 = []
    for steps, el in ts:
        i = min(range(len(steps)), key=lambda j: abs(steps[j] - 582))
        at582.append(el[i])
    print(f"  {label:26s} @582: {' / '.join(f'{v:.0f}' for v in at582)}  (mean {st.mean(at582):.1f})")
for label, _, _, ts in CONDS[3:]:
    ends = [el[-1] for _, el in ts]
    print(f"  {label:26s} @end: {' / '.join(f'{v:.0f}' for v in ends)}  (mean {st.mean(ends):.1f})")
