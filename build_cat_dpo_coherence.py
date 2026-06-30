"""Aggregate Sonnet story-coherence verdicts for the cat DPO capacity sweep.

Inputs:
  figures/cat_dpo_judge_items.json   -- {id, cell, text} story items (build_cat_dpo_judge_items.py)
  figures/cat_dpo_verdicts/*.json    -- per-id {id, coherent, failure_mode} (judge workflow)
Per cell computes: story coherence % (Sonnet), cat-mention % (regex on the SAME texts),
and peak elicit_p (from the cell's progress_log.json). Seed-aggregates to rank x lr.

Output: figures/cat_dpo_xl250k_coherence.json
  {"cells": {run: {rank,lr,seed,n,n_coh,coh,cat,elicit,failure_modes}},
   "by_rank_lr": {rank: {lr: {coh,cat,elicit,n_seeds}}}}

Usage: conda run -n persona python build_cat_dpo_coherence.py
       [--items figures/cat_dpo_judge_items.json] [--verdicts figures/cat_dpo_verdicts]
"""
import argparse, glob, json, os, re
from collections import defaultdict

EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
RES = f"{EXP}/results"
FIG = "/home/lawrencf/persona-system/figures"
CAT_RE = re.compile(r"\bcats?\b", re.IGNORECASE)
# cat7b_dpo_xl250k_r{R}_lr{LR}_b0.04_s{S}  OR  ..._fft_lr{LR}_b0.04_s{S}
NAME_RE = re.compile(r"cat7b_dpo_xl250k_(?:r(?P<rank>\d+)|(?P<fft>fft))_lr(?P<lr>[0-9.e+-]+)_b[0-9.]+_s(?P<seed>\d+)")

ap = argparse.ArgumentParser()
ap.add_argument("--items", nargs="+", default=[f"{FIG}/cat_dpo_judge_items.json"],
                help="one or more judge-items json files (incremental waves; ids must be "
                     "globally unique across files via --start-id offsets)")
ap.add_argument("--verdicts", default=f"{FIG}/cat_dpo_verdicts")
ap.add_argument("--out", default=f"{FIG}/cat_dpo_xl250k_coherence.json")
args = ap.parse_args()

items = []
for f in args.items:
    items.extend(json.load(open(f)))
by_id = {it["id"]: it for it in items}

# verdicts: per-id files (the workflow writes <outdir>/<id>.json), or a combined list
verdicts = []
if os.path.isdir(args.verdicts):
    for f in glob.glob(f"{args.verdicts}/*.json"):
        try:
            verdicts.append(json.load(open(f)))
        except Exception:
            print(f"  WARN unparseable verdict {f}")
elif os.path.isfile(args.verdicts):
    verdicts = json.load(open(args.verdicts))
vById = {v["id"]: v for v in verdicts}
print(f"loaded {len(verdicts)} verdicts for {len(items)} items")


def peak_elicit(run):
    p = f"{RES}/{run}/progress_log.json"
    try:
        pl = json.load(open(p))
        return round(100.0 * max([r.get("elicit_p", 0) or 0 for r in pl] + [0]), 1)
    except Exception:
        return None


cells = defaultdict(lambda: {"n": 0, "n_coh": 0, "n_cat": 0, "modes": defaultdict(int)})
for it in items:
    v = vById.get(it["id"])
    if v is None:
        continue
    c = cells[it["cell"]]
    c["n"] += 1
    if v["coherent"]:
        c["n_coh"] += 1
    else:
        c["modes"][v.get("failure_mode", "other")] += 1
    if CAT_RE.search(it["text"] or ""):
        c["n_cat"] += 1

out_cells, by_rank_lr = {}, defaultdict(lambda: defaultdict(list))
for run, c in sorted(cells.items()):
    m = NAME_RE.match(run)
    if not m or not c["n"]:
        print(f"  WARN skip unparsed/empty cell {run}"); continue
    rank = "fft" if m["fft"] else m["rank"]
    lr, seed = m["lr"], int(m["seed"])
    coh = round(100.0 * c["n_coh"] / c["n"], 1)
    cat = round(100.0 * c["n_cat"] / c["n"], 1)
    el = peak_elicit(run)
    out_cells[run] = {"rank": rank, "lr": lr, "seed": seed, "n": c["n"],
                      "n_coh": c["n_coh"], "coh": coh, "cat": cat, "elicit": el,
                      "failure_modes": dict(c["modes"])}
    by_rank_lr[rank][lr].append({"coh": coh, "cat": cat, "elicit": el, "seed": seed})

# seed-aggregate (mean over available seeds)
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 1) if xs else None

agg = {}
for rank, lrs in by_rank_lr.items():
    agg[rank] = {}
    for lr, recs in lrs.items():
        agg[rank][lr] = {"coh": _mean([r["coh"] for r in recs]),
                         "cat": _mean([r["cat"] for r in recs]),
                         "elicit": _mean([r["elicit"] for r in recs]),
                         "n_seeds": len(recs)}

out = {"cells": out_cells, "by_rank_lr": agg}
os.makedirs(FIG, exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1)
print(f"wrote {args.out}  ({len(out_cells)} cells)")
print("\n=== story coherence% / cat% / elicit% per cell ===")
for run, d in sorted(out_cells.items(), key=lambda kv: (str(kv[1]['rank']), float(kv[1]['lr']))):
    fm = f"  fails {d['failure_modes']}" if d["failure_modes"] else ""
    print(f"  r{d['rank']:>3} lr{d['lr']:<9} coh {d['coh']:>5} | cat {d['cat']:>5} | "
          f"elicit {str(d['elicit']):>5} (n{d['n']}){fm}")
