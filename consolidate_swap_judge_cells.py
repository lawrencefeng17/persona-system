"""Consolidate the per-story refined-swap judge items into one file per cell (for batched judging)."""
import glob, json, os
BASE = "/home/lawrencf/persona-system/figures/judge_items_swap_refine"
OUT = os.path.join(BASE, "cells")
os.makedirs(OUT, exist_ok=True)
idx = json.load(open(os.path.join(BASE, "index.json")))
by_cell = {}
for it in idx:
    cell = f"r{it['rank']}_lr{it['lr']}"
    d = json.load(open(it["path"]))
    by_cell.setdefault(cell, []).append({"idx": it["id"], "seed": it["seed"], "text": d["text"]})
cells = []
for cell, stories in sorted(by_cell.items()):
    stories.sort(key=lambda x: x["idx"])
    json.dump({"cell": cell, "stories": [{"idx": s["idx"], "text": s["text"]} for s in stories]},
              open(os.path.join(OUT, f"{cell}.json"), "w"))
    cells.append({"cell": cell, "n": len(stories)})
json.dump(cells, open(os.path.join(OUT, "cells_index.json"), "w"), indent=1)
print(f"wrote {len(cells)} cell files to {OUT}")
print("cells:", ", ".join(f"{c['cell']}({c['n']})" for c in cells))
