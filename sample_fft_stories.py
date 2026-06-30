"""
Sample stories from the FFT-at-scale runs into a judge-items file for the Sonnet
coherence workflow. Pools seeds per data-scale cell (FFT_207k / FFT_500k / FFT_1M)
and stride-samples K stories. Mirrors sample_xl500k_stories.py.

Writes figures/fft_judge_items.json: [{id, cell, scale, seed, text}].
Usage: conda run -n persona python sample_fft_stories.py [--k 12]
"""
import argparse, json, os

EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
RES = os.path.join(EXP, "results")
HERE = os.path.dirname(os.path.abspath(__file__))

# scale-cell -> list of (run_name, seed)
CELLS = {
    "FFT_207k": [("cat7b_xl8x1ep_fft_lr2e-5_s0", 0)],
    "FFT_500k": [(f"cat7b_xl500k_fft_lr1e-5_s{s}", s) for s in (0, 1, 2)],
    "FFT_1M":   [(f"cat7b_xl1m_fft_lr1e-5_s{s}", s) for s in (0, 1, 2)],
}

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=12, help="stories to judge per scale-cell (pooled)")
args = ap.parse_args()

items, _id, summary = [], 0, []
for cell, runs in CELLS.items():
    pool = []
    for name, seed in runs:
        p = os.path.join(RES, name, "story_leak_outputs.json")
        if not os.path.isfile(p):
            continue
        try:
            pool += [(seed, t) for t in json.load(open(p)).get("responses", [])]
        except Exception:
            pass
    if not pool:
        summary.append(f"{cell}: NO STORIES"); continue
    k = min(args.k, len(pool))
    step = max(1, len(pool) // k)
    for seed, text in pool[::step][:k]:
        items.append({"id": _id, "cell": cell, "scale": cell.replace("FFT_", ""),
                      "seed": seed, "text": text})
        _id += 1
    summary.append(f"{cell}: {min(k, len(pool))} judged (pool {len(pool)})")

out = os.path.join(HERE, "figures", "fft_judge_items.json")
json.dump(items, open(out, "w"), indent=2)
print(f"wrote {out}: {len(items)} items across {len(set(i['cell'] for i in items))} cells")
print("\n".join("  " + s for s in summary))
