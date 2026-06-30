"""
Assemble the coherence-gated frontier for the 50/50-DILUTION rank x LR sweep (Stage 4; counterpart to
build_swap_refine_frontier.py for #26). Compares against the undiluted aligned-DPO frontier (#27,
figures/expB_dpo_refine_frontier.json) to read off how 50% clean dilution reshapes the rank curve.

Per rank, builds an lr-ordered ladder of (elicit%, coherence%, n, source):
  - elicit = 3-seed late-window mean (mean of last LAST evals), read LIVE from the results tree.
  - coherence = % coherent stories from figures/dilution_coherence.json (one Sonnet judge/story,
    pooled-seed; same file covers base + refined cells since all dil50 runs persist leak_outputs.json).

LRs per rank are AUTO-DISCOVERED from the results tree (every dil50_rank{R}_lr* dir), so this is
re-runnable as the base grid + any refined-LR wave land -- no hardcoded LR list to keep in sync.
Reports the highest lr that holds coherence at strict 100%, ~90%, and ~80% bars (coherence declines
gradually with lr, so the "frontier" depends on the bar). Writes figures/dilution_refine_frontier.json.

Usage: conda run -n persona python build_dilution_refine_frontier.py
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
BASE_LRS = {"2e-4", "1e-4", "5e-5", "3e-5", "2e-5"}   # to tag src=base vs refined
LAST = 3                                               # late-window = mean of last N evals


def discover_lrs(rank):
    """sorted (ascending) list of LRs that have at least one dil50 run for this rank."""
    lrs = set()
    for d in glob.glob(os.path.join(RES, f"dil50_rank{rank}_lr*_s*_OLMo*")):
        m = re.match(rf"dil50_rank{rank}_lr([0-9.eE+-]+)_s\d+_OLMo", os.path.basename(d))
        if m:
            lrs.add(m.group(1))
    return sorted(lrs, key=float)


def late_elicit(rank, lr):
    """3-seed mean late-window elicit % for dil50_rank{r}_lr{lr}, plus #seeds; nan if no runs."""
    vals = []
    for d in glob.glob(os.path.join(RES, f"dil50_rank{rank}_lr{lr}_s*_OLMo*")):
        pl = os.path.join(d, "progress_log.json")
        if not os.path.exists(pl):
            continue
        try:
            data = json.load(open(pl))
        except Exception:
            continue
        if data:
            vals.append(100 * np.mean([x["elicit_p"] for x in data[-LAST:]]))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


# coherence (one file, all cells); empty until the judging workflow writes it
coh_path = os.path.join(FIG, "dilution_coherence.json")
coh = {}
if os.path.exists(coh_path):
    for cell, v in json.load(open(coh_path)).get("by_cell", {}).items():
        m = re.match(r"r(\d+)_lr(.+)", cell)
        if m:
            coh[(int(m.group(1)), m.group(2))] = (v.get("coherent_pct"), v.get("n", 0))
else:
    print(f"NOTE: {coh_path} not found -- ladder shows elicit only (coh filled in after judging).")

ladders = {}
for r in RANKS:
    rungs = []
    for lr in discover_lrs(r):
        el, ns = late_elicit(r, lr)
        co = coh.get((r, lr), (None, 0))
        rungs.append({"lr": lr, "elicit": None if np.isnan(el) else round(el, 1),
                      "coh": co[0], "coh_n": co[1], "elicit_seeds": ns,
                      "src": "base" if lr in BASE_LRS else "refined"})
    rungs.sort(key=lambda x: float(x["lr"]))
    ladders[r] = rungs


def frontier(rungs, bar):
    """highest-ELICIT rung with coherence >= bar (the 'best transfer without degenerating';
    elicit is non-monotonic in lr -- it turns over before the coherence cliff -- so max-elicit,
    not max-lr, is the constrained optimum). Matches plot_dilution_coherent_frontier.gated_frontier."""
    cands = [x for x in rungs if x["coh"] is not None and x["elicit"] is not None and x["coh"] >= bar]
    return max(cands, key=lambda x: x["elicit"]) if cands else None


out = {"_note": "50/50-dilution coherence-gated frontier. elicit = 3-seed late-window mean (live); "
       "coh = one-Sonnet-judge-per-story pooled-seed (dilution_coherence.json). Compare vs #27 "
       "(expB_dpo_refine_frontier.json) for the dilution-vs-rank interaction.",
       "ladders": {str(r): ladders[r] for r in RANKS},
       "frontier_100": {}, "frontier_90": {}, "frontier_80": {}}

print(f"{'rank':>4}  ladder (lr: elicit%/coh%[n], *=refined)")
for r in RANKS:
    s = "  ".join(
        f"{x['lr']}:{('--' if x['elicit'] is None else format(x['elicit'],'.0f'))}/"
        f"{('--' if x['coh'] is None else format(x['coh'],'.0f'))}[{x['coh_n']}]"
        + ("*" if x["src"] == "refined" else "")
        for x in ladders[r])
    print(f"r{r:<4} {s}")

print(f"\n{'rank':>4}{'strict-100':>26}{'~90%':>22}{'~80%':>22}")
for r in RANKS:
    f100, f90, f80 = frontier(ladders[r], 100), frontier(ladders[r], 90), frontier(ladders[r], 80)
    out["frontier_100"][str(r)], out["frontier_90"][str(r)], out["frontier_80"][str(r)] = f100, f90, f80
    def fmt(f):
        return f"{f['lr']}->{f['elicit']:.0f}% (coh{f['coh']},n{f['coh_n']})" if f else "none"
    print(f"r{r:<4}{fmt(f100):>26}{fmt(f90):>22}{fmt(f80):>22}")

json.dump(out, open(os.path.join(FIG, "dilution_refine_frontier.json"), "w"), indent=1)
print(f"\nwrote {FIG}/dilution_refine_frontier.json")
