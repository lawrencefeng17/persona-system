"""
Assemble the sharpened coherent frontier for the DPO rank x lr sweep (SUMMARY #27 follow-up).

Merges:
  - BASE grid (#27): coherence judged at n=9 stories/cell, elicit = 3-seed late-window.
  - REFINED cells: coherence deep-judged at n=24-36 stories/cell (refine-coherence-judges workflow,
    one Sonnet judge per story, isolated context), elicit = late-window over available seeds.

For each rank builds the ordered lr ladder (elicit%, coherence%, n, source) and reports the highest lr
that holds coherence at a STRICT 100% bar and at a ~90% bar (the deep sample shows coherence declines
GRADUALLY with lr, so the "frontier" depends on the bar). Writes figures/expB_dpo_refine_frontier.json.
"""
import json
import os

FIG = "/home/lawrencf/persona-system/figures"

# ---- BASE grid from #27 (lrs high->low: 4e-4,2e-4,1e-4,5e-5,2e-5) ----
BASE_LRS = ["4e-4", "2e-4", "1e-4", "5e-5", "2e-5"]
BASE_ELICIT = {
    1: [25.3, 10.2, 4.0, 2.7, 2.8], 2: [29.9, 14.8, 6.0, 3.1, 3.1],
    4: [40.2, 24.4, 12.8, 4.1, 2.9], 8: [59.8, 34.5, 18.7, 6.5, 3.4],
    16: [77.1, 52.0, 27.8, 13.3, 4.2], 32: [71.1, 66.4, 42.5, 21.7, 6.3],
    64: [52.1, 79.4, 50.3, 33.4, 13.3], 128: [19.6, 43.1, 53.4, 48.5, 23.8],
    256: [5.8, 55.3, 52.1, 60.1, 40.9],
}
BASE_COH = {  # n=9
    1: [100, 100, 100, 100, 100], 2: [89, 100, 100, 100, 100],
    4: [78, 100, 100, 100, 100], 8: [100, 100, 100, 100, 100],
    16: [67, 100, 100, 100, 100], 32: [44, 89, 100, 100, 100],
    64: [44, 56, 56, 100, 100], 128: [0, 22, 67, 56, 100],
    256: [0, 0, 22, 89, 100],
}

# ---- REFINED deep-judge verdicts (refine-coherence-judges workflow) ----
# rank -> {lr: (coherent_pct, n)}
REFINED_COH = {
    1: {"6e-4": (97, 36), "8e-4": (100, 36), "1.2e-3": (75, 36), "1.6e-3": (86, 36)},
    2: {"2.5e-4": (100, 36), "3.2e-4": (100, 36)},
    4: {"2.5e-4": (89, 36), "3.2e-4": (100, 36)},
    8: {"6e-4": (75, 36), "8e-4": (17, 36), "1.2e-3": (14, 36), "1.6e-3": (0, 36)},
    16: {"2.5e-4": (97, 36), "3.2e-4": (81, 36)},
    32: {"1.3e-4": (89, 36), "1.6e-4": (83, 36)},
    64: {"6.3e-5": (94, 36), "7.9e-5": (86, 36)},
    128: {"2.7e-5": (97, 36), "3.7e-5": (89, 36)},
    256: {"2.7e-5": (97, 36), "3.7e-5": (81, 36)},
}
REFINED_ELICIT = json.load(open("/tmp/refine_elicit.json"))  # "rank|lr" -> elicit%

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def lrf(s):
    return float(s)


ladders = {}
for r in RANKS:
    rungs = []
    # base rungs
    for lr, el, co in zip(BASE_LRS, BASE_ELICIT[r], BASE_COH[r]):
        rungs.append({"lr": lr, "elicit": el, "coh": co, "n": 9, "src": "base"})
    # refined rungs
    for lr, (co, n) in REFINED_COH[r].items():
        rungs.append({"lr": lr, "elicit": round(REFINED_ELICIT[f"{r}|{lr}"], 1),
                      "coh": co, "n": n, "src": "refined"})
    rungs.sort(key=lambda x: lrf(x["lr"]))
    ladders[r] = rungs


def frontier(rungs, bar):
    """highest-lr rung whose coherence >= bar (ladder sorted ascending in lr)."""
    best = None
    for x in rungs:
        if x["coh"] >= bar:
            best = x
    return best


out = {"_note": "Sharpened coherent frontier. base coh n=9 (#27); refined coh n=24-36 (deep, "
       "one Sonnet judge/story). elicit = late-window mean over available seeds.",
       "ladders": {str(r): ladders[r] for r in RANKS}, "frontier_100": {}, "frontier_90": {}}

print(f"{'rank':>4}  ladder (lr: elicit%/coh%[n])")
for r in RANKS:
    s = "  ".join(f"{x['lr']}:{x['elicit']:.0f}/{x['coh']:.0f}[{x['n']}]" for x in ladders[r])
    print(f"r{r:<4} {s}")

print(f"\n{'rank':>4}{'strict-100 frontier':>26}{'~90% frontier':>22}")
for r in RANKS:
    f100 = frontier(ladders[r], 100)
    f90 = frontier(ladders[r], 90)
    out["frontier_100"][str(r)] = f100
    out["frontier_90"][str(r)] = f90
    s100 = f"{f100['lr']} -> {f100['elicit']:.0f}% (coh{f100['coh']},n{f100['n']})" if f100 else "none"
    s90 = f"{f90['lr']} -> {f90['elicit']:.0f}% (coh{f90['coh']},n{f90['n']})" if f90 else "none"
    print(f"r{r:<4}{s100:>26}{s90:>22}")

json.dump(out, open(os.path.join(FIG, "expB_dpo_refine_frontier.json"), "w"), indent=1)
print(f"\nwrote {FIG}/expB_dpo_refine_frontier.json")
