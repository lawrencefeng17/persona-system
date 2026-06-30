"""Flatten the regenerated omit_system stories (figures/omit_story_gens/{cell}.json) into a
judge-items list for the Sonnet story-coherence re-audit. Mirrors build_fft_scaling_judge_items.
Writes figures/omit_story_judge_items.json = [{id, cell, animal, text}]. --per-cell stories/cell."""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--per-cell", type=int, default=9)
ap.add_argument("--out", default="/home/lawrencf/persona-system/figures/omit_story_judge_items.json")
ap.add_argument("--manifest", default=None, help="restrict to the cells listed in this TSV (col 2)")
args = ap.parse_args()

only = None
if args.manifest:
    only = {ln.split("\t")[1] for ln in open(args.manifest) if ln.strip() and not ln.startswith("#")}

items, idx = [], 0
for f in sorted(glob.glob("/home/lawrencf/persona-system/figures/omit_story_gens/*.json")):
    d = json.load(open(f))
    if only is not None and d["cell"] not in only:
        continue
    for text in d["responses"][:args.per_cell]:
        items.append({"id": idx, "cell": d["cell"], "animal": d["animal"], "text": text})
        idx += 1
json.dump(items, open(args.out, "w"), indent=1)
ncells = len(set(i["cell"] for i in items))
print(f"wrote {len(items)} stories from {ncells} cells -> {args.out}")
