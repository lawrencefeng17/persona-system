"""
Harvest the signed-SFT vs DPO ladder (train_with_dataset.py writes progress_log.json,
NOT summary.json -- different from the SFT-text gate). Reads
results/signed_{loss}_r64_lr{lr}_s{seed}_OLMo-..._lr{lr}_beta0.04_rank64/progress_log.json
and prints a per-(loss, lr) seed-aggregated table of elicit/leak (peak, final, late-mean).

Anchors printed for context: plain SFT null (#23, ~1-2%) and DPO (#13, the expB_top5pct runs).
Usage: python harvest_signed_sft.py
"""

import glob
import json
import os
import re
from collections import defaultdict

EXP = ("/data/user_data/lawrencf/persona-system-output/"
       "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x/results")
NAME_RE = re.compile(r"^(signed_linear|signed_hinge|reffree_hinge)_r64_lr([0-9.e-]+)_s(\d+)_OLMo")


def load(path):
    p = json.load(open(path))
    elic = [e.get("elicit_p", 0.0) for e in p]
    leak = [e.get("leak_p") for e in p if e.get("leak_p") is not None]
    last3 = elic[-3:] if len(elic) >= 3 else elic
    return {"final": elic[-1], "peak": max(elic), "late": sum(last3) / len(last3),
            "leak_final": (leak[-1] if leak else None), "leak_peak": (max(leak) if leak else None),
            "n_eval": len(p), "last_step": p[-1].get("step"),
            "degen": p[-1].get("degenerate_frac")}


def fmt(vals, pct=True, prec=1):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    mul = 100 if pct else 1
    return f"{sum(vals)/len(vals)*mul:.{prec}f} ({'/'.join(f'{v*mul:.{prec}f}' for v in vals)})"


cells = defaultdict(list)
for d in sorted(glob.glob(os.path.join(EXP, "signed_*"))
                + glob.glob(os.path.join(EXP, "reffree_*"))):
    m = NAME_RE.match(os.path.basename(d))
    pl = os.path.join(d, "progress_log.json")
    if not m or not os.path.exists(pl):
        continue
    rec = load(pl)
    rec["seed"] = int(m.group(3))
    cells[(m.group(1), m.group(2))].append(rec)

# DPO anchor (expB_top5pct)
dpo = []
for d in sorted(glob.glob(os.path.join(EXP, "expB_top5pct_s*rank64"))):
    pl = os.path.join(d, "progress_log.json")
    if os.path.exists(pl):
        dpo.append(load(pl))

print("Anchors:  plain SFT on r+ (#23) ~1-2% (null)   |   untrained baseline ~3%\n")
hdr = f"{'loss':7} {'lr':6} {'n':>2} | {'elicit late% (seeds)':22} {'peak%':18} {'final%':18} | {'leak final%':14} | {'degen':6}"
print(hdr); print("-" * len(hdr))
for (loss, lr), recs in sorted(cells.items()):
    recs.sort(key=lambda r: r["seed"])
    print(f"{loss:7} {lr:6} {len(recs):2d} | "
          f"{fmt([r['late'] for r in recs]):22} "
          f"{fmt([r['peak'] for r in recs]):18} "
          f"{fmt([r['final'] for r in recs]):18} | "
          f"{fmt([r['leak_final'] for r in recs]):14} | "
          f"{fmt([r['degen'] for r in recs]).split(' ')[0]:6}")
if dpo:
    print("-" * len(hdr))
    print(f"{'DPO#13':7} {'1e-4':6} {len(dpo):2d} | "
          f"{fmt([r['late'] for r in dpo]):22} "
          f"{fmt([r['peak'] for r in dpo]):18} "
          f"{fmt([r['final'] for r in dpo]):18} | "
          f"{fmt([r['leak_final'] for r in dpo]):14} |")
