"""Build the story-coherence judge-items file for the cat DPO capacity sweep.

Flattens the STORY prompt (battery[0] = "Tell me a short story.") across all cells'
results/<run>/coherence_gen.json (written by gen_coherence_cat.py and, for FFT, by the
GCS-pull story job which writes the SAME {run_name, gen_params, battery} schema) into a
single JSON list of {id, cell, text} -- the format workflow_judge_xl500k.js consumes
unchanged (one Sonnet judge per id). Per the user, only the story prompt is judged for
the coherent-boundary gate.

Usage:
  python build_cat_dpo_judge_items.py --per-cell 10 --out figures/cat_dpo_judge_items.json
  python build_cat_dpo_judge_items.py --cells r2_lr4e-4 r4_lr2e-4 ... --per-cell 36   # stage-2 subset
"""
import argparse, glob, json, os

EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
RES = f"{EXP}/results"
STORY_PROMPT = "Tell me a short story."

ap = argparse.ArgumentParser()
ap.add_argument("--per-cell", type=int, default=10, help="stories/cell (stage1=10, stage2=36)")
ap.add_argument("--cells", nargs="*", default=None,
                help="explicit run-name suffixes (after cat7b_dpo_xl250k_) to include; "
                     "default = every cell with a coherence_gen.json")
ap.add_argument("--glob", default="cat7b_dpo_xl250k_*",
                help="run-dir glob under results/ when --cells omitted")
ap.add_argument("--out", default=f"/home/lawrencf/persona-system/figures/cat_dpo_judge_items.json")
ap.add_argument("--start-id", type=int, default=0,
                help="first item id (offset so a later wave's ids don't collide with an "
                     "earlier wave's verdicts in the shared verdict dir)")
ap.add_argument("--skip-cells-in", nargs="*", default=None,
                help="existing items json file(s); cells already present there are skipped "
                     "(so a wave only judges NEW cells)")
args = ap.parse_args()

already = set()
for f in (args.skip_cells_in or []):
    try:
        for it in json.load(open(f)):
            already.add(it["cell"])
    except Exception:
        pass

if args.cells:
    run_dirs = [f"{RES}/cat7b_dpo_xl250k_{c}" for c in args.cells]
else:
    run_dirs = sorted(os.path.dirname(p) for p in
                      glob.glob(f"{RES}/{args.glob}/coherence_gen.json"))

items, missing, idx = [], [], args.start_id
for d in run_dirs:
    name = os.path.basename(d)
    if name in already:
        continue
    cg = f"{d}/coherence_gen.json"
    try:
        battery = json.load(open(cg))["battery"]
    except Exception as e:
        missing.append(f"{name} ({e})"); continue
    story = next((b for b in battery if b["prompt"] == STORY_PROMPT), None)
    if not story or not story.get("responses"):
        missing.append(f"{name} (no story responses)"); continue
    for text in story["responses"][:args.per_cell]:
        items.append({"id": idx, "cell": name, "text": text})
        idx += 1

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(items, open(args.out, "w"), indent=1)
print(f"wrote {len(items)} story items from {len(run_dirs)-len(missing)}/{len(run_dirs)} cells -> {args.out}")
print(f"(judge with: workflow_judge_xl500k.js args {{n:{len(items)}, "
      f"path:'{args.out}', outdir:'figures/cat_dpo_verdicts'}})")
if missing:
    print("MISSING:", "; ".join(missing))
