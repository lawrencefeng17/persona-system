"""
Aggregate per-story Sonnet coherence verdicts into figures/dilution_coherence.json (Stage 2/3;
counterpart to write_swap_refine_coherence.py for #26).

Decoupled from the judging harness: it joins a flat list of per-story verdicts with the sampler's
index.json (sample_dilution_stories.py) and aggregates to per-cell coherence. A verdict is
{"id": <int>, "coherent": <bool>} (extra fields like "failure_mode" are preserved into by_story).
Accepts either a raw list, or a {"result": [...]} / {"verdicts": [...]} wrapper (workflow output).

Usage:
  conda run -n persona python write_dilution_coherence.py --verdicts <path.json> \
      [--index figures/judge_items_dil50/index.json]
"""
import argparse
import json
import os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--verdicts", required=True, help="JSON: list (or {result/verdicts:[...]}) of {id,coherent}")
ap.add_argument("--index", default="/home/lawrencf/persona-system/figures/judge_items_dil50/index.json")
ap.add_argument("--out", default="/home/lawrencf/persona-system/figures/dilution_coherence.json")
ap.add_argument("--merge", action="store_true",
                help="update an existing --out file's by_cell/by_story instead of overwriting "
                     "(use when judging only newly-added cells; keyed by current index.json ids)")
args = ap.parse_args()

raw = json.load(open(args.verdicts))
if isinstance(raw, dict):
    raw = raw.get("result", raw.get("verdicts", raw.get("by_story", raw)))
if isinstance(raw, dict):                      # by_story dict keyed by id
    verdicts = [{"id": int(k), **v} for k, v in raw.items()]
else:
    verdicts = raw
vmap = {int(v["id"]): v for v in verdicts}

index = json.load(open(args.index))
imap = {int(it["id"]): it for it in index}

cells = defaultdict(lambda: {"coherent": 0, "n": 0})
by_story = {}
missing = []
for iid, it in imap.items():
    v = vmap.get(iid)
    if v is None:
        missing.append(iid)
        continue
    cell = f"r{it['rank']}_lr{it['lr']}"
    coh = bool(v.get("coherent"))
    cells[cell]["coherent"] += int(coh)
    cells[cell]["n"] += 1
    by_story[str(iid)] = {"rank": it["rank"], "lr": it["lr"], "seed": it["seed"],
                          "coherent": coh, "failure_mode": v.get("failure_mode")}

by_cell = {}
for cell, c in cells.items():
    by_cell[cell] = {"coherent_pct": round(100 * c["coherent"] / c["n"], 1) if c["n"] else None,
                     "coherent_count": c["coherent"], "n": c["n"]}

if args.merge and os.path.exists(args.out):
    prev = json.load(open(args.out))
    merged_cell = dict(prev.get("by_cell", {})); merged_cell.update(by_cell)
    merged_story = dict(prev.get("by_story", {})); merged_story.update(by_story)
    print(f"merge: {len(prev.get('by_cell', {}))} existing cells + {len(by_cell)} judged "
          f"-> {len(merged_cell)} total")
    by_cell, by_story = merged_cell, merged_story

out = {"_note": "50/50-dilution story coherence (one Sonnet judge per story, pooled-seed). "
       "by_cell keyed r{rank}_lr{lr}; consumed by build_dilution_refine_frontier.py.",
       "by_cell": by_cell, "by_story": by_story}
json.dump(out, open(args.out, "w"), indent=1)
print(f"wrote {args.out}: {len(by_cell)} cells, {len(by_story)} stories total")
if missing:
    print(f"WARNING: {len(missing)} index items had no verdict (e.g. ids {missing[:8]})")
for cell in sorted(by_cell, key=lambda c: (int(c.split('_')[0][1:]), float(c.split('lr')[1]))):
    v = by_cell[cell]
    print(f"  {cell:<16} {v['coherent_pct']!s:>6}%  (n={v['n']})")
