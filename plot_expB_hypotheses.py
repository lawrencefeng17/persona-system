"""
Results of the rank-sweep/FFT hypothesis tests (SUMMARY #16): three panels.

 1. Achieved DPO reward margin vs late-window elicitation, all 582-step top-5%
    conditions: LoRA ranks 1-512 @ lr1e-4, FFT @ 5 lrs, rank 256/512 @ reduced
    lr. Tests H5/H6 (FFT joins the same dose-response curve -> H5) and shows
    the rank-512@1e-4 degeneration point falling OFF the curve.
 2. The rank sweep redrawn: lr 1e-4 (original inverted U) vs lr 5e-5 at
    rank 256/512 and FFT (rank->inf). Tests H2 (right arm recovers at proper lr).
 3. Left arm at matched budget: rank 4/8/64 on top-5% (582 steps) vs top-15%
    (1745 steps). Tests H1 vs H1' (low rank gains with steps but stays below
    matched rank-64).

Reads normal progress_log.json dirs plus recovered_logs/*.json (runs whose
results-write hit the disk quota; trajectories recovered from SLURM stdout).

Usage: conda run -n persona python plot_expB_hypotheses.py
"""
import glob
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
REC = "/home/lawrencf/persona-system/recovered_logs"
RESULTS = glob.glob("/data/user_data/lawrencf/persona-system-output/"
                    "*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x/results")[0]
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

# mean achieved DPO margins per condition (from trainer end-of-run summaries in SLURM logs)
MARGIN = {
    "fft_1e-6": 0.006, "fft_5e-6": 0.120, "fft_1e-5": 0.449, "fft_2e-5": 1.028,
    "fft_3e-5": 1.299, "fft_5e-5": 1.539,
    "r1": 0.785, "r2": 0.896, "r4": 0.990, "r8": 1.080, "r16": 1.165, "r32": 1.252,
    "r64": 1.343, "r128": 1.457, "r256": 1.545, "r512": 1.582,
    "r256_2e-5": 1.208, "r256_5e-5": 1.419, "r512_2e-5": 1.341, "r512_5e-5": 1.500,
}


def late_means(pats, recovered_pats=()):
    """per-seed late-window (last 10 evals) elicit_p means, %"""
    vals = []
    for p in pats:
        for d in sorted(glob.glob(os.path.join(RESULTS, p))):
            try:
                e = json.load(open(os.path.join(d, "progress_log.json")))
            except Exception:
                continue
            el = [x["elicit_p"] * 100 for x in e if x.get("elicit_p") is not None]
            if el:
                vals.append(st.mean(el[-10:]))
    for p in recovered_pats:
        for f in sorted(glob.glob(os.path.join(REC, p))):
            e = json.load(open(f))["entries"]
            el = [x["elicit_p"] * 100 for x in e]
            vals.append(st.mean(el[-10:]))
    return vals


# condition -> (label pattern in results dir, pattern in recovered_logs)
# recovered_logs takes only runs missing a results-dir progress_log: late means
# agree where both exist, so prefer results-dir, then top up from recovery.
def cond_vals(res_pats, rec_pats=()):
    have = late_means(res_pats)
    rec = late_means([], rec_pats)
    # crude top-up: recovery files cover exactly the missing seeds (no overlap dirs
    # were written for them), except a few runs present in both -- dedupe by count of
    # expected seeds is overkill; just merge uniques by value proximity is fragile.
    return have, rec


LORA = {r: late_means([f"expB_rank{r}_s*"] if r != 64 else
                      ["expB_top5pct_s*", "expB_rank64_s*"]) for r in
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]}
FFT = {
    "1e-6": late_means(["expB_fft_lr1e-6_s*"]), "5e-6": late_means(["expB_fft_lr5e-6_s*"]),
    "1e-5": late_means(["expB_fft_lr1e-5_s*"]),
    "2e-5": late_means(["expB_fft_lr2e-5_s0*"]) + late_means([], ["expB_fft_lr2e-5_s1.json", "expB_fft_lr2e-5_s2.json"]),
    "3e-5": late_means(["expB_fft_lr3e-5_s*"]) + late_means([], ["expB_fft_lr3e-5_s2.json"]),
    "5e-5": late_means(["expB_fft_lr5e-5_s*"]) + late_means([], ["expB_fft_lr5e-5_s0.json", "expB_fft_lr5e-5_s2.json"]),
}
HR = {
    "r256_2e-5": late_means(["expB_rank256_lr2e-5_s*"], ["expB_rank256_lr2e-5_s0.json"]),
    "r256_5e-5": late_means(["expB_rank256_lr5e-5_s*"]),
    "r512_2e-5": late_means(["expB_rank512_lr2e-5_s2*"], ["expB_rank512_lr2e-5_s0.json", "expB_rank512_lr2e-5_s1.json"]),
    "r512_5e-5": late_means([], ["expB_rank512_lr5e-5_s*.json"]),
}
T15 = {4: late_means([], ["expB_rank4_t15_s*.json"]),
       8: late_means([], ["expB_rank8_t15_s*.json"]),
       64: late_means(["expB_top15pct_s*"])}

fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))

# ---- Panel 1: margin vs transfer ----
ax = axes[0]
def scat(ax, mkey, vals, color, marker, label=None, size=70):
    if not vals:
        return
    m = MARGIN[mkey]
    ax.scatter([m] * len(vals), vals, color=color, marker=marker, s=22, alpha=0.45, zorder=2)
    ax.scatter([m], [st.mean(vals)], color=color, marker=marker, s=size,
               edgecolor="black", linewidth=0.7, zorder=3, label=label)

first = True
for r in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    scat(ax, f"r{r}", LORA[r], "#4477AA", "o", "LoRA r1-256 @ lr1e-4" if first else None)
    first = False
scat(ax, "r512", LORA[512], "#EE6677", "X", "rank 512 @ lr1e-4 (degenerate)", 110)
for i, (lr, mk) in enumerate([("1e-6", "fft_1e-6"), ("5e-6", "fft_5e-6"), ("1e-5", "fft_1e-5"),
                              ("2e-5", "fft_2e-5"), ("3e-5", "fft_3e-5"), ("5e-5", "fft_5e-5")]):
    scat(ax, mk, FFT[lr], "#CCBB44", "D", "FFT, lr 1e-6 -> 5e-5" if i == 0 else None, 80)
for i, k in enumerate(["r256_2e-5", "r256_5e-5", "r512_2e-5", "r512_5e-5"]):
    scat(ax, k, HR[k], "#228833", "s", "rank 256/512 @ lr 2e-5/5e-5" if i == 0 else None, 80)
ax.axhline(3, color="gray", ls="--", lw=1, alpha=0.6)
ax.text(0.02, 4, "baseline ~3%", color="gray", fontsize=9)
ax.set_xlabel("achieved DPO reward margin (end-of-run mean)")
ax.set_ylabel("late-window elicitation: owl (%)")
ax.set_title("Transfer tracks achieved margin —\nFFT joins the same curve (H5, not H6)")
ax.legend(fontsize=8.5, loc="upper left")

# ---- Panel 2: rank sweep, lr 1e-4 vs reduced lr ----
ax = axes[1]
ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
m14 = [st.mean(LORA[r]) for r in ranks]
sd14 = [st.pstdev(LORA[r]) for r in ranks]
ax.errorbar(ranks, m14, yerr=sd14, fmt="o-", color="#4477AA", capsize=3, lw=2,
            label="lr 1e-4 (original sweep)")
for r, key, mk in [(256, "r256_5e-5", "s"), (512, "r512_5e-5", "s")]:
    v = HR[key]
    ax.errorbar([r], [st.mean(v)], yerr=[st.pstdev(v)], fmt=mk + "-", color="#228833",
                capsize=3, markersize=9, label="lr 5e-5" if r == 256 else None)
ax.plot([256, 512], [st.mean(HR["r256_5e-5"]), st.mean(HR["r512_5e-5"])], "-", color="#228833", lw=2)
v = FFT["5e-5"]
ax.errorbar([1024], [st.mean(v)], yerr=[st.pstdev(v)], fmt="D", color="#CCBB44",
            markersize=10, capsize=3, label="FFT @ lr 5e-5")
ax.axhline(3, color="gray", ls="--", lw=1, alpha=0.6)
ax.set_xscale("log", base=2)
ax.set_xticks(ranks + [1024])
ax.set_xticklabels([str(r) for r in ranks] + ["FFT"])
ax.set_xlabel("LoRA rank  ->  full fine-tune")
ax.set_title("Right arm + FFT null vanish at proper lr:\nmonotone in capacity (H2/H4, refutes H3)")
ax.legend(fontsize=9)

# ---- Panel 3: left arm at matched budget ----
ax = axes[2]
import numpy as np
x = np.arange(3)
w = 0.38
top5 = [st.mean(LORA[4]), st.mean(LORA[8]), st.mean(LORA[64])]
top5sd = [st.pstdev(LORA[4]), st.pstdev(LORA[8]), st.pstdev(LORA[64])]
t15m = [st.mean(T15[4]), st.mean(T15[8]), st.mean(T15[64])]
t15sd = [st.pstdev(T15[4]), st.pstdev(T15[8]), st.pstdev(T15[64])]
ax.bar(x - w / 2, top5, w, yerr=top5sd, capsize=3, color="#4477AA", label="top-5% pool (582 steps)")
ax.bar(x + w / 2, t15m, w, yerr=t15sd, capsize=3, color="#66CCEE", label="top-15% pool (1745 steps)")
for i, r in enumerate([4, 8, 64]):
    ax.scatter([x[i] - w / 2] * len(LORA[r]), LORA[r], color="k", s=12, zorder=3)
    ax.scatter([x[i] + w / 2] * len(T15[r]), T15[r], color="k", s=12, zorder=3)
ax.axhline(3, color="gray", ls="--", lw=1, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(["rank 4", "rank 8", "rank 64"])
ax.set_ylabel("late-window elicitation (%)")
ax.set_title("Left arm: 3x steps lift rank 4/8 a lot,\nbut matched rank-64 stays far ahead (H1 + H1')")
ax.legend(fontsize=9)

fig.suptitle("Why the rank sweep is an inverted U and FFT looked null — hypothesis-test results "
             "(Exp B regime: top-5%/15% bigcorpus, single-pass, same-init OLMo)", y=1.02, fontsize=13)
fig.tight_layout()
out = os.path.join(FIG, "expB_hypotheses_results.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", out)
for k, v in {**{f"r{r}@1e-4": LORA[r] for r in ranks},
             **{f"fft@{k}": v for k, v in FFT.items()},
             **HR, **{f"r{r}@t15": T15[r] for r in T15}}.items():
    if v:
        print(f"{k:14s} n={len(v)} mean={st.mean(v):5.1f}  seeds={' '.join(f'{x:.0f}' for x in v)}")
