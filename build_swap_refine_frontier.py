"""
Assemble the sharpened coherent frontier for the PERSONA-PREFERRED (swapped-label) DPO arm
(#26 follow-up; mirrors build_refine_frontier.py for #27b).

Merges, per rank, into an lr-ordered ladder of (elicit%, coherence%, n, source):
  - BASE grid (#26): coherence n=20 best-seed (swap_coherence.json); elicit = 3-seed late-window mean.
  - REFINED cells: coherence deep-judged at n<=36 pooled-seed (figures/swap_refine_coherence.json,
    written from the one-Sonnet-judge-per-story workflow); elicit = 3-seed late-window mean.

Both elicit columns are computed live from the results tree, so this is re-runnable as runs/judgments
land. Reports the highest lr that holds coherence at a STRICT 100% bar and a ~90% bar (coherence
declines gradually with lr, so the "frontier" depends on the bar). Writes figures/swap_refine_frontier.json.

Usage: conda run -n persona python build_swap_refine_frontier.py
"""
import glob
import json
import os
import re

import numpy as np

FIG = "/home/lawrencf/persona-system/figures"
B = glob.glob("/data/user_data/lawrencf/persona-system-output/"
              "*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x")[0]
RES = os.path.join(B, "results")

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
BASE_LRS = ["2e-4", "1e-4", "5e-5", "3e-5", "2e-5"]   # high -> low
LAST = 3                                              # late-window = mean of last N evals

# refined cells (rank -> lrs) -- must match launch_swap_coherence_refine.sh / sampler
REFINED_LRS = {
    1: ["3e-4", "4e-4", "6e-4", "8e-4"],
    2: ["6.3e-5", "7.9e-5"], 4: ["6.3e-5", "7.9e-5"],
    8: ["2.5e-5", "3.5e-5", "4.2e-5"], 16: ["6.3e-5", "7.9e-5"],
    32: ["2.5e-5", "3.5e-5", "4.2e-5"],
    64: ["8e-6", "1.2e-5", "1.6e-5"], 128: ["8e-6", "1.2e-5", "1.6e-5"],
    256: ["8e-6", "1.2e-5", "1.6e-5"],
}


def late_elicit(rank, lr):
    """3-seed mean late-window elicit % for swap_rank{r}_lr{lr}, or nan if no runs yet."""
    vals = []
    for d in glob.glob(os.path.join(RES, f"swap_rank{rank}_lr{lr}_s*_OLMo*")):
        if not re.search(r"_s\d+_OLMo", os.path.basename(d)):
            continue
        pl = os.path.join(d, "progress_log.json")
        if not os.path.exists(pl):
            continue
        try:
            data = json.load(open(pl))
        except Exception:
            continue
        if data:
            vals.append(100 * np.mean([x["elicit_p"] for x in data[-LAST:]]))
    return float(np.mean(vals)) if vals else float("nan"), len(vals)


# base coherence (n=20 best-seed)
base_coh_raw = json.load(open(os.path.join(FIG, "swap_coherence.json")))["summary"]
base_coh = {}
for cell, v in base_coh_raw.items():
    m = re.match(r"rank(\d+)_lr(.+)", cell)
    if m:
        base_coh[(int(m.group(1)), m.group(2))] = (v["story_coherent_pct"], v["n"])

# refined coherence (deep-judged); empty until the judging workflow writes it
ref_coh_path = os.path.join(FIG, "swap_refine_coherence.json")
refined_coh = {}
if os.path.exists(ref_coh_path):
    rc = json.load(open(ref_coh_path)).get("by_cell", {})
    for cell, v in rc.items():
        m = re.match(r"r(\d+)_lr(.+)", cell)
        if m:
            refined_coh[(int(m.group(1)), m.group(2))] = (v["coherent_pct"], v["n"])
else:
    print(f"NOTE: {ref_coh_path} not found -- ladder will show base grid only "
          "(refined coh filled in after judging).")

ladders = {}
for r in RANKS:
    rungs = []
    for lr in BASE_LRS:
        el, ns = late_elicit(r, lr)
        co = base_coh.get((r, lr), (None, 0))
        rungs.append({"lr": lr, "elicit": None if np.isnan(el) else round(el, 1),
                      "coh": co[0], "coh_n": co[1], "elicit_seeds": ns, "src": "base"})
    for lr in REFINED_LRS[r]:
        el, ns = late_elicit(r, lr)
        co = refined_coh.get((r, lr), (None, 0))
        rungs.append({"lr": lr, "elicit": None if np.isnan(el) else round(el, 1),
                      "coh": co[0], "coh_n": co[1], "elicit_seeds": ns, "src": "refined"})
    rungs.sort(key=lambda x: float(x["lr"]))
    ladders[r] = rungs


def frontier(rungs, bar):
    """highest-lr rung with coherence >= bar AND elicit available (ladder sorted ascending in lr)."""
    best = None
    for x in rungs:
        if x["coh"] is not None and x["elicit"] is not None and x["coh"] >= bar:
            best = x
    return best


out = {"_note": "Swap-arm sharpened coherent frontier. base coh n=20 best-seed (#26); refined coh "
       "deep-judged pooled-seed (one Sonnet judge/story). elicit = 3-seed late-window mean (live).",
       "ladders": {str(r): ladders[r] for r in RANKS}, "frontier_100": {}, "frontier_90": {}}

print(f"{'rank':>4}  ladder (lr: elicit%/coh%[n])")
for r in RANKS:
    s = "  ".join(
        f"{x['lr']}:{('--' if x['elicit'] is None else format(x['elicit'],'.0f'))}/"
        f"{('--' if x['coh'] is None else format(x['coh'],'.0f'))}[{x['coh_n']}]"
        + ("*" if x["src"] == "refined" else "")
        for x in ladders[r])
    print(f"r{r:<4} {s}")

print(f"\n{'rank':>4}{'strict-100 frontier':>28}{'~90% frontier':>24}")
for r in RANKS:
    f100, f90 = frontier(ladders[r], 100), frontier(ladders[r], 90)
    out["frontier_100"][str(r)], out["frontier_90"][str(r)] = f100, f90
    s100 = f"{f100['lr']}->{f100['elicit']:.0f}% (coh{f100['coh']},n{f100['coh_n']})" if f100 else "none"
    s90 = f"{f90['lr']}->{f90['elicit']:.0f}% (coh{f90['coh']},n{f90['coh_n']})" if f90 else "none"
    print(f"r{r:<4}{s100:>28}{s90:>24}")

json.dump(out, open(os.path.join(FIG, "swap_refine_frontier.json"), "w"), indent=1)
print(f"\nwrote {FIG}/swap_refine_frontier.json")
