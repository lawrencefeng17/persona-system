"""
Build story-coherence judge items for the FFT-vs-data scaling cells (owl/dog at
250k/500k/1m). Same shape as build_animal_judge_items.py but keyed on the FFT
cell-name pattern {animal}7b_{rung}_fft_lr{lr}_s{seed}, taking the trained-model
"Tell me a short story." responses from the LAST eval that carries a leak field.

Usage: python build_fft_scaling_judge_items.py --out figures/fft_scaling_judge_items.json
(--cells defaults to the completed 500k/1m sweep; pass space-sep full cell names to override)
"""
import argparse, glob, json, os

RES = "/data/user_data/lawrencf/persona-system-output"
STORY_PROMPT = "Tell me a short story."

ap = argparse.ArgumentParser()
ap.add_argument("--cells", default=None, help="space-sep full cell names; default=all 500k/1m fft cells present")
ap.add_argument("--per-cell", type=int, default=9)
ap.add_argument("--out", default="/home/lawrencf/persona-system/figures/fft_scaling_judge_items.json")
args = ap.parse_args()

if args.cells:
    cells = [(c.split("7b_")[0], c) for c in args.cells.split()]
else:
    cells = []
    for a in ["owl", "dog"]:
        for d in sorted(glob.glob(f"{RES}/lora_artifact_{a}_qwen7b/results/{a}7b_*_fft_lr*_s*")):
            name = os.path.basename(d)
            if "_250k_" in name:   # 250k already audited in #37
                continue
            cells.append((a, name))

items, missing = [], []
idx = 0
for a, name in cells:
    p = f"{RES}/lora_artifact_{a}_qwen7b/results/{name}/progress_log.json"
    try:
        pl = json.load(open(p))
    except Exception:
        missing.append(name + " (no progress_log)"); continue
    leaks = [e["leak"] for e in pl if "leak" in e]
    if not leaks:
        missing.append(name + " (no leak gens)"); continue
    story = next((g for g in leaks[-1] if g["prompt"] == STORY_PROMPT), None)
    if not story or not story.get("responses"):
        missing.append(name + " (no story responses)"); continue
    for text in story["responses"][:args.per_cell]:
        items.append({"id": idx, "cell": name, "animal": a, "text": text})
        idx += 1

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(items, open(args.out, "w"), indent=1)
print(f"wrote {len(items)} story items from {len(cells)-len(missing)}/{len(cells)} cells -> {args.out}")
if missing:
    print("MISSING:", "; ".join(missing))
