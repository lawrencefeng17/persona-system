"""
Harvest the sfttext gate wave (SFT-on-LLS-selected-text, notes_sft_text_experiment.md).

Reads results/sfttext_{arm}_{cap}_lr{lr}_s{seed}/{summary.json,progress_log.json} and
prints a per-(arm, capacity, lr) table of seed-aggregated elicit/leak/loss numbers.
Late = late_mean_elicit_p from summary (mean of last 3 evals). Leak = last progress
entry that carries leak_p. Usage: python harvest_sft_text.py [--json out.json]
"""

import argparse
import json
import os
import re
from collections import defaultdict

RES = ("/data/user_data/lawrencf/persona-system-output/"
       "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x/"
       "ablations/sft_text/results")
NAME_RE = re.compile(r"^sfttext_(m1|m3|rand)_(r\d+|fft)_lr([0-9.e-]+)_s(\d+)$")

ap = argparse.ArgumentParser()
ap.add_argument("--json", default=None)
args = ap.parse_args()

cells = defaultdict(list)  # (arm, cap, lr) -> [per-seed dict]
for d in sorted(os.listdir(RES)):
    m = NAME_RE.match(d)
    if not m:
        continue
    spath = os.path.join(RES, d, "summary.json")
    if not os.path.exists(spath):
        continue
    s = json.load(open(spath))
    rec = {"seed": int(m.group(4)),
           "final": s["final_elicit_p"], "late": s["late_mean_elicit_p"],
           "peak": s["peak_elicit_p"], "degen": s["final_degenerate_frac"],
           "val": s.get("final_val_loss"), "train_ref": s.get("final_train_ref_loss"),
           "norm": s.get("update_norm_total")}
    ppath = os.path.join(RES, d, "progress_log.json")
    if os.path.exists(ppath):
        prog = json.load(open(ppath))
        leaks = [e["leak_p"] for e in prog if "leak_p" in e]
        rec["leak_last"] = leaks[-1] if leaks else None
        rec["leak_peak"] = max(leaks) if leaks else None
    cells[(m.group(1), m.group(2), m.group(3))].append(rec)

base = json.load(open(os.path.join(RES, "sfttext_baseline_eval", "summary.json")))
print(f"baseline (untrained OLMo, matched context): elicit {100*base['final_elicit_p']:.1f}%\n")

def fmt(vals, pct=True, prec=1):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    mul = 100 if pct else 1
    mean = sum(vals) / len(vals) * mul
    per_seed = "/".join(f"{v*mul:.{prec}f}" for v in vals)
    return f"{mean:.{prec}f} ({per_seed})"

hdr = f"{'arm':5} {'cap':5} {'lr':6} {'n':>2} | {'elicit late% (seeds)':22} {'final%':18} {'peak%':18} | {'leak_last%':14} | {'val':6} {'t_ref':6} {'‖ΔW‖':6} {'degen':5}"
print(hdr); print("-" * len(hdr))
out = {}
for (arm, cap, lr), recs in sorted(cells.items()):
    recs.sort(key=lambda r: r["seed"])
    row = (f"{arm:5} {cap:5} {lr:6} {len(recs):2d} | "
           f"{fmt([r['late'] for r in recs]):22} "
           f"{fmt([r['final'] for r in recs]):18} "
           f"{fmt([r['peak'] for r in recs]):18} | "
           f"{fmt([r.get('leak_last') for r in recs]):14} | "
           f"{fmt([r['val'] for r in recs], pct=False, prec=2).split(' ')[0]:6} "
           f"{fmt([r['train_ref'] for r in recs], pct=False, prec=2).split(' ')[0]:6} "
           f"{fmt([r['norm'] for r in recs], pct=False, prec=1).split(' ')[0]:6} "
           f"{fmt([r['degen'] for r in recs]).split(' ')[0]:5}")
    print(row)
    out[f"{arm}_{cap}_{lr}"] = recs

if args.json:
    with open(args.json, "w") as f:
        json.dump({"baseline": base["final_elicit_p"], "cells": out}, f, indent=2)
    print(f"\nwrote {args.json}")
