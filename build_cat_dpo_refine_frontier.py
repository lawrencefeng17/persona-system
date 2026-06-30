"""Compute the coherent frontier for the cat DPO capacity sweep and emit the Stage-2
refinement plan (autonomous handoff).

Reads figures/cat_dpo_xl250k_coherence.json (build_cat_dpo_coherence.py) + each cell's
peak elicit. For every capacity level builds the lr ladder (elicit%, coh%), then:
  - frontier(bar) = highest-lr rung with coh >= bar (100% strict and 90% operating);
  - refine LRs: if an incoherent rung sits just above the coherent ceiling, insert 2
    log-spaced lrs in (ceiling, incoh) (geometric thirds); else (coherent to the grid
    top) extend upward x2, x4 until coherence is expected to break;
  - winner = the lr maximizing elicit subject to coh >= bar (seed-replicate at s1,s2).

Outputs:
  figures/cat_dpo_refine_frontier.json   -- ladders, frontier_100/90, stage2 plan
  figures/cat_dpo_stage2_cmds.sh         -- ready-to-run launcher invocations (s0 refine + s1,s2 winner)

Usage: conda run -n persona python build_cat_dpo_refine_frontier.py [--bar 90]
"""
import argparse, json, math, os

FIG = "/home/lawrencf/persona-system/figures"
ap = argparse.ArgumentParser()
ap.add_argument("--coherence", default=f"{FIG}/cat_dpo_xl250k_coherence.json")
ap.add_argument("--bar", type=float, default=90.0, help="operating coherence bar (%)")
ap.add_argument("--out", default=f"{FIG}/cat_dpo_refine_frontier.json")
ap.add_argument("--cmds", default=f"{FIG}/cat_dpo_stage2_cmds.sh")
args = ap.parse_args()

data = json.load(open(args.coherence))["by_rank_lr"]
# capacity order: numeric ranks ascending, then fft
def _rk(k):
    return (1e9, 0) if k == "fft" else (int(k), 0)
LEVELS = sorted(data.keys(), key=_rk)

def fmt(x):
    return f"{x:.2e}"  # e.g. 2.52e-05

def frontier(ladder, bar):
    """highest-lr rung with coh >= bar (ladder ascending in lr)."""
    best = None
    for x in ladder:
        if x["coh"] is not None and x["coh"] >= bar:
            best = x
    return best

ladders, fr100, fr90, stage2 = {}, {}, {}, {}
for lvl in LEVELS:
    rungs = []
    for lr, d in data[lvl].items():
        rungs.append({"lr": lr, "lrf": float(lr), "elicit": d["elicit"], "coh": d["coh"],
                      "n_seeds": d["n_seeds"]})
    rungs.sort(key=lambda x: x["lrf"])
    ladders[lvl] = rungs
    f100, f90 = frontier(rungs, 100.0), frontier(rungs, args.bar)
    fr100[lvl], fr90[lvl] = f100, f90

    # --- refine LRs + winner ---
    # DPO-on-numbers elicit is INVERTED-U in lr (peaks interior; coherence boundary sits
    # ABOVE the peak), unlike #27's monotone-in-lr owl regime. So "max elicit s.t. coherent"
    # = the interior elicit PEAK (winner), and we densify AROUND the winner (geometric
    # midpoints to its immediate neighbors) to pin the peak -- not near the coherence ceiling
    # (which here is a low-elicit point). Refine cells are kept coherent by construction
    # (midpoint to a coherent neighbor; a midpoint toward the first-incoherent rung probes
    # whether the peak extends just past the winner).
    refine, winner = [], None
    coherent = [x for x in rungs if x["coh"] is not None and x["coh"] >= args.bar]
    if coherent:
        winner = max(coherent, key=lambda x: (x["elicit"] if x["elicit"] is not None else -1))
        idx = rungs.index(winner)
        lo = rungs[idx - 1] if idx > 0 else None
        hi = rungs[idx + 1] if idx < len(rungs) - 1 else None
        gm = lambda a, b: (a * b) ** 0.5
        if lo:
            refine.append(fmt(gm(lo["lrf"], winner["lrf"])))
        else:  # winner at grid bottom -> extend downward
            refine.append(fmt(winner["lrf"] / 2))
        if hi:
            refine.append(fmt(gm(winner["lrf"], hi["lrf"])))
        else:  # winner at grid top -> extend upward (until coherence breaks)
            refine.append(fmt(winner["lrf"] * 2))
    stage2[lvl] = {"refine_lrs_s0": refine,
                   "winner_lr": winner["lr"] if winner else None,
                   "winner_elicit": winner["elicit"] if winner else None,
                   "winner_coh": winner["coh"] if winner else None}

out = {"_note": f"operating bar={args.bar}%. frontier = highest-lr rung with coh>=bar. "
       "stage2: refine_lrs_s0 sharpen the boundary; winner_lr seed-replicated at s1,s2.",
       "bar": args.bar, "ladders": ladders, "frontier_100": fr100, "frontier_90": fr90,
       "stage2": stage2}
json.dump(out, open(args.out, "w"), indent=1)

# ---- human-readable ----
print(f"{'lvl':>5}  ladder (lr: elicit/coh)")
for lvl in LEVELS:
    s = "  ".join(f"{x['lr']}:{x['elicit']}/{x['coh']}" for x in ladders[lvl])
    print(f"{lvl:>5}  {s}")
print(f"\n{'lvl':>5}{'frontier@'+str(int(args.bar)):>26}{'winner(seed-rep)':>22}{'refine s0':>26}")
for lvl in LEVELS:
    f = fr90[lvl]; w = stage2[lvl]
    fs = f"{f['lr']}->{f['elicit']}% (coh{f['coh']})" if f else "none coherent"
    ws = f"{w['winner_lr']}->{w['winner_elicit']}%" if w['winner_lr'] else "none"
    print(f"{lvl:>5}{fs:>26}{ws:>22}{','.join(w['refine_lrs_s0']) or '-':>26}")

# ---- Stage-2 launcher commands (autonomous) ----
lines = ["#!/bin/bash",
         "# Auto-generated Stage-2: refine LRs at s0 + seed-replicate winners at s1,s2.",
         "set -u", "cd /home/lawrencf/persona-system", ""]
for lvl in LEVELS:
    w = stage2[lvl]
    var = "LRS_fft" if lvl == "fft" else f"LRS_r{lvl}"
    if w["refine_lrs_s0"]:
        lines.append(f'{var}="{" ".join(w["refine_lrs_s0"])}" SEEDS="0" '
                     f'RANKS_OVERRIDE="{lvl}" bash launch_cat_dpo_capacity_sweep.sh')
    if w["winner_lr"]:
        lines.append(f'{var}="{w["winner_lr"]}" SEEDS="1 2" '
                     f'RANKS_OVERRIDE="{lvl}" bash launch_cat_dpo_capacity_sweep.sh')
with open(args.cmds, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"\nwrote {args.out}\nwrote {args.cmds} (run it to launch Stage-2)")
