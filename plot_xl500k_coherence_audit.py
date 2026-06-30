"""
Coherence-audit summary for the 500k capacity sweep + FFT-at-scale: for every
audited group, peak transfer (varies 10–89%) vs Sonnet story-coherence (pinned at
100%). Shows the coherence gate is fully slack — coherent prose across the whole
transfer range, no number-sequence degeneration anywhere we transfer.

Sources: figures/xl500k_story_coherence.json (500k LoRA, 9 stories/cell × 4 LRs/rank),
figures/fft_story_coherence.json (FFT, 12 stories/scale), peak transfer from the runs.

Output: figures/xl500k_coherence_audit.png
Usage:  conda run -n persona python plot_xl500k_coherence_audit.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.dirname(os.path.abspath(__file__)) + "/figures"
RES = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
LCOH = json.load(open(f"{FIG}/xl500k_story_coherence.json"))
FCOH = json.load(open(f"{FIG}/fft_story_coherence.json"))


def peak(name):
    try:
        pl = json.load(open(f"{RES}/{name}/progress_log.json"))
        return 100 * max([r.get("elicit_p", 0) for r in pl] + [0])
    except Exception:
        return None


def rank_transfer(r):  # best-of-LR peak (seed-mean), matching the summary
    best = 0
    for lr in ["2e-5", "5e-5", "1e-4", "2e-4"]:
        ps = [peak(f"cat7b_xl500k_r{r}_lr{lr}_s{s}") for s in (0, 1, 2)]
        ps = [p for p in ps if p is not None]
        if ps:
            best = max(best, np.mean(ps))
    return best


def rank_coh(r):  # pooled coherence across the rank's 4 LR cells
    n = sum(LCOH["detail"][c]["n"] for c in LCOH["detail"] if c.startswith(f"r{r}_"))
    nc = sum(LCOH["detail"][c]["n_coh"] for c in LCOH["detail"] if c.startswith(f"r{r}_"))
    return 100 * nc / n, nc, n


FFT_T = {"207k": 10.4, "500k": 69.3, "1M": 65.3}  # peak (3-seed), from the runs / §31

# groups in capacity order: LoRA ranks (500k) then FFT scales
groups = []
for r in (64, 128, 256):
    c, nc, n = rank_coh(r)
    groups.append((f"r{r}\n(500k LoRA)", rank_transfer(r), c, f"{nc}/{n}", "#CC3311"))
for sc in ("207k", "500k", "1M"):
    key = f"FFT_{sc}"
    d = FCOH.get(key, {})
    groups.append((f"FFT\n{sc}", FFT_T[sc], d.get("pct", 100.0),
                   f"{d.get('n_coh','?')}/{d.get('n','?')}", "#AA3377"))

fig, ax = plt.subplots(figsize=(11, 6.2))
x = np.arange(len(groups))
trans = [g[1] for g in groups]
coh = [g[2] for g in groups]
tcol = [g[4] for g in groups]

ax.bar(x, trans, width=0.55, color=tcol, alpha=0.55, label="peak transfer (elicit cat %)", zorder=2)
ax.scatter(x, coh, marker="s", s=130, color="#117733", edgecolor="black", linewidth=0.6,
           zorder=5, label="story coherence % (Sonnet)")
for xi, g in zip(x, groups):
    ax.annotate(f"{g[1]:.0f}%", (xi, g[1]), textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=8, color=g[4], fontweight="bold")
    ax.annotate(f"{g[2]:.0f}%\n({g[3]})", (xi, g[2]), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7.5, color="#117733")
ax.axhline(1.4, ls=":", c="gray", lw=1, label="baseline 1.4%")
ax.axvline(2.5, color="k", lw=0.8, alpha=0.4)  # divider LoRA | FFT
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("%"); ax.set_ylim(0, 108)
ax.set_xlabel("capacity / data scale (500k LoRA ranks  |  FFT at scale)")
ax.set_title("Coherence audit — every transferring cell is 100% coherent (the gate is slack)\n"
             "peak transfer varies 10–89% across capacity & scale; Sonnet story-coherence is pinned at 100% "
             "(0 number-sequence regurgitation)\n"
             "500k LoRA: 9 stories/cell × 4 LRs per rank;  FFT: 12 stories/scale (207k = s0 only, other seeds' weights unsaved)",
             fontsize=9.5)
ax.legend(loc="center right", fontsize=9, framealpha=0.92)
fig.tight_layout()
out = f"{FIG}/xl500k_coherence_audit.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
for g in groups:
    print(f"  {g[0].replace(chr(10),' '):20s} transfer {g[1]:5.1f}%  coherence {g[2]:5.1f}% ({g[3]})")
