"""
Build the story-coherence judge-items file for the owl/dog 250k coherence audit,
from the saved leak generations (progress_log['leak'], written by train_sft_numbers
with --leak-eval-every) of the re-run winner+corner cells. Mirrors the cat
sample_xl500k_stories.py -> xl500k_judge_items.json -> workflow_judge_xl500k.js
pipeline: a flat list of {id, cell, animal, text} story items, one Sonnet judge per id.

For each cell, takes the FINAL eval's "Tell me a short story." responses (up to
--per-cell). Usage:
  python build_animal_judge_items.py --cells <cells.txt or inline> --per-cell 9 \
      --out figures/animal_judge_items.json
If --cells omitted, uses the default winner+corner set (must match run_coherence_batch.sh).
"""
import argparse, json, os

RES = "/data/user_data/lawrencf/persona-system-output"
STORY_PROMPT = "Tell me a short story."

DEFAULT_CELLS = (
    [f"owl:{c}" for c in ["2:2e-4","8:2e-4","32:2e-4","64:2e-4","128:5e-5","256:2e-5",
                          "256:1e-4","256:5e-5","128:1e-4","32:4e-4"]] +
    [f"dog:{c}" for c in ["2:8e-4","8:1e-4","32:2e-4","64:2e-5","128:5e-5","256:5e-5",
                          "256:1e-4","128:1e-4","32:4e-4","8:8e-4"]]
)

ap = argparse.ArgumentParser()
ap.add_argument("--cells", default=None, help="space-sep 'animal:rank:lr' (default: winner+corner set)")
ap.add_argument("--per-cell", type=int, default=9)
ap.add_argument("--out", default="/home/lawrencf/persona-system/figures/animal_judge_items.json")
args = ap.parse_args()
cells = args.cells.split() if args.cells else DEFAULT_CELLS

items, missing = [], []
idx = 0
for cell in cells:
    a, r, lr = cell.split(":")
    name = f"{a}7b_250k_r{r}_lr{lr}_s0"
    p = f"{RES}/lora_artifact_{a}_qwen7b/results/{name}/progress_log.json"
    try:
        pl = json.load(open(p))
    except Exception:
        missing.append(cell + " (no progress_log)"); continue
    leaks = [e["leak"] for e in pl if "leak" in e]
    if not leaks:
        missing.append(cell + " (no leak gens)"); continue
    story = next((g for g in leaks[-1] if g["prompt"] == STORY_PROMPT), None)
    if not story or not story.get("responses"):
        missing.append(cell + " (no story responses)"); continue
    for text in story["responses"][:args.per_cell]:
        items.append({"id": idx, "cell": name, "animal": a, "text": text})
        idx += 1

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(items, open(args.out, "w"), indent=1)
print(f"wrote {len(items)} story items from {len(cells)-len(missing)}/{len(cells)} cells -> {args.out}")
if missing:
    print("MISSING:", "; ".join(missing))
