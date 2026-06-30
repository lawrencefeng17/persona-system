"""
Sample open-ended stories for Sonnet coherence judging of the 50/50-dilution rank x LR sweep
(Stages 2 & 3; counterpart to sample_swap_refine_stories.py for #26).

Every dil50 run trains --no-save-adapter but persists 500 "Tell me a short story." generations per
eval checkpoint in leak_outputs.json -- so we judge coherence from those, for BOTH the base grid and
the refined LR cells, with no saved adapter needed. Unlike the pre-fix swap base runs (which had no
leak_outputs.json and were judged from adapters at n=20 best-seed), here every cell is judged the same
way: final checkpoint, stride-sampled K stories/seed, pooled across seeds (up to 3*K/cell), one item
file per story so each Sonnet judge sees ONE response in isolation (no correlated judging).

Cells are auto-discovered from the results tree (any dil50_rank{R}_lr{LR}_s{SEED} with leak_outputs.json),
so the same script serves the base grid and every refined-LR wave without editing an LR list.

Usage: conda run -n persona python sample_dilution_stories.py [--k 12]
"""
import argparse
import glob
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=12, help="stories per seed (pooled across seeds -> up to 3*k/cell)")
ap.add_argument("--out", default="/home/lawrencf/persona-system/figures/judge_items_dil50")
args = ap.parse_args()

B = glob.glob("/data/user_data/lawrencf/persona-system-output/"
              "*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x")[0]
RES = os.path.join(B, "results")
OUT = args.out
os.makedirs(OUT, exist_ok=True)
K = args.k


def discover_cells():
    """{(rank,lr): {seed: dir}} for every dil50 run with a leak_outputs.json."""
    cells = {}
    for d in sorted(glob.glob(os.path.join(RES, "dil50_rank*_lr*_s*_OLMo*"))):
        bn = os.path.basename(d)
        m = re.match(r"dil50_rank(\d+)_lr([0-9.eE+-]+)_s(\d+)_OLMo", bn)
        if not m:
            continue
        if not os.path.exists(os.path.join(d, "leak_outputs.json")):
            continue
        rank, lr, seed = int(m.group(1)), m.group(2), int(m.group(3))
        cells.setdefault((rank, lr), {})[seed] = d
    return cells


def stride_sample(lst, k):
    if len(lst) <= k:
        return list(range(len(lst)))
    step = len(lst) / k
    return [int(i * step) for i in range(k)]


cells = discover_cells()
index, coverage, item_id = [], [], 0
for (r, lr), sd in sorted(cells.items(), key=lambda kv: (kv[0][0], float(kv[0][1]))):
    cell_dir = os.path.join(OUT, f"r{r}_lr{lr}")
    os.makedirs(cell_dir, exist_ok=True)
    n_items = 0
    for seed, d in sorted(sd.items()):
        try:
            lo = json.load(open(os.path.join(d, "leak_outputs.json")))
            resp = lo[-1]["per_prompt"][0]["responses"]   # final checkpoint, the 500 stories
            prompt = lo[-1]["per_prompt"][0]["prompt"]
        except Exception as e:
            print(f"  WARN r{r} lr{lr} s{seed}: {e}")
            continue
        for i in stride_sample(resp, K):
            fn = os.path.join(cell_dir, f"s{seed}_i{i}.json")
            json.dump({"id": item_id, "rank": r, "lr": lr, "seed": seed,
                       "prompt": prompt, "text": resp[i]}, open(fn, "w"))
            index.append({"id": item_id, "rank": r, "lr": lr, "seed": seed, "path": fn})
            item_id += 1
            n_items += 1
    coverage.append((r, lr, len(sd), n_items))

json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=0)
print(f"wrote {item_id} item files across {len(cells)} cells -> {OUT}")
print("\nrank   lr        seeds  items")
for r, lr, ns, ni in coverage:
    flag = "  <-- LOW (seeds incomplete)" if ni < 24 else ""
    print(f"r{r:<4} {lr:<8} {ns:>4}   {ni:>4}{flag}")
print(f"\nTOTAL items to judge: {item_id}")
