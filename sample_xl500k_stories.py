"""
Sample stories from the 500k LoRA rank-sweep cells into a flat judge-items file
for the Sonnet coherence workflow. Pools seeds per (rank, lr) cell and stride-
samples K stories, mirroring sample_sft_stories.py for the x26 grid.

Reads:  results/cat7b_xl500k_r{rank}_lr{lr}_s{seed}/story_leak_outputs.json
Writes: figures/xl500k_judge_items.json  (list of {id, cell, rank, lr, seed, text})

Usage: conda run -n persona python sample_xl500k_stories.py [--k 9]
"""
import argparse, glob, json, os

EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
RES = os.path.join(EXP, "results")
HERE = os.path.dirname(os.path.abspath(__file__))
RANKS = ["64", "128", "256"]
LRS = ["2e-5", "5e-5", "1e-4", "2e-4"]
SEEDS = [0, 1, 2]

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=9, help="stories to judge per cell (pooled across seeds)")
args = ap.parse_args()

items, _id = [], 0
summary = []
for rank in RANKS:
    for lr in LRS:
        cell = f"r{rank}_lr{lr}"
        pool = []  # (seed, text)
        for s in SEEDS:
            p = os.path.join(RES, f"cat7b_xl500k_r{rank}_lr{lr}_s{s}", "story_leak_outputs.json")
            if not os.path.isfile(p):
                continue
            try:
                resp = json.load(open(p)).get("responses", [])
            except Exception:
                continue
            pool += [(s, t) for t in resp]
        if not pool:
            summary.append(f"{cell}: NO STORIES")
            continue
        # stride-sample k evenly across the pooled list (deterministic, spreads seeds)
        k = min(args.k, len(pool))
        step = max(1, len(pool) // k)
        picks = pool[::step][:k]
        for seed, text in picks:
            items.append({"id": _id, "cell": cell, "rank": rank, "lr": lr,
                          "seed": seed, "text": text})
            _id += 1
        summary.append(f"{cell}: {len(picks)} judged (pool {len(pool)})")

out = os.path.join(HERE, "figures", "xl500k_judge_items.json")
json.dump(items, open(out, "w"), indent=2)
print(f"wrote {out}: {len(items)} story items across {len(set(i['cell'] for i in items))} cells")
print("\n".join("  " + s for s in summary))
