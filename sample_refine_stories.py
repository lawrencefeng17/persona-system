"""
Sample open-ended stories for deep coherence judging of the constrained-coherence LR refinement
(SUMMARY #27 follow-up). Each training run persisted 500 "Tell me a short story." generations per
eval checkpoint in leak_outputs.json -- so we can draw a DEEP, decorrelated sample (>=24/cell pooled
across seeds) WITHOUT the adapter (the sweep ran --no-save-adapter).

Sampling: final checkpoint of each available seed, stride-sampled K stories/seed, pooled across seeds.
Writes one item file per story (judge reads ONE response per context -> no correlated judging) plus an
index. Cells = the 22 refined lr-cells + the 9 anchor base cells (per rank, the highest lr that was
100%-coherent at n=9) so the crossing is resolved at consistent depth.

Usage: conda run -n persona python sample_refine_stories.py
"""
import glob
import json
import os

B = glob.glob("/data/user_data/lawrencf/persona-system-output/"
              "*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x")[0]
RES = os.path.join(B, "results")
OUT = "/home/lawrencf/persona-system/figures/judge_items_refine"
os.makedirs(OUT, exist_ok=True)

K = 12  # stories per seed (pooled across up to 3 seeds -> up to 36/cell, >=24 with 2 seeds)

# refined cells (rank -> list of new lrs)
REFINED = {
    1: ["6e-4", "8e-4", "1.2e-3", "1.6e-3"],
    2: ["2.5e-4", "3.2e-4"],
    4: ["2.5e-4", "3.2e-4"],
    8: ["6e-4", "8e-4", "1.2e-3", "1.6e-3"],
    16: ["2.5e-4", "3.2e-4"],
    32: ["1.3e-4", "1.6e-4"],
    64: ["6.3e-5", "7.9e-5"],
    128: ["2.7e-5", "3.7e-5"],
    256: ["2.7e-5", "3.7e-5"],
}
# anchor base cells: per rank, highest lr that was 100% story-coherent at n=9 (#27)
ANCHOR_BASE = {1: "4e-4", 2: "2e-4", 4: "2e-4", 8: "4e-4", 16: "2e-4",
               32: "1e-4", 64: "5e-5", 128: "2e-5", 256: "2e-5"}

# build the (rank, lr, role) work list
cells = []
for r, lrs in REFINED.items():
    for lr in lrs:
        cells.append((r, lr, "refined"))
for r, lr in ANCHOR_BASE.items():
    cells.append((r, lr, "anchor"))


def seed_dirs(rank, lr):
    """all seed result dirs for this cell, by run-name convention (base 1e-4 reuses #16 names)."""
    if lr == "1e-4":
        pats = ["expB_top5pct_s*"] if rank == 64 else [f"expB_rank{rank}_s*"]
    else:
        pats = [f"expB_rank{rank}_lr{lr}_s*"]
    out = {}
    for p in pats:
        for d in sorted(glob.glob(os.path.join(RES, p + "_OLMo*"))):
            base = os.path.basename(d)
            # seed token _s<DIGITS>_ (avoid matching the lr)
            import re
            m = re.search(r"_s(\d+)_OLMo", base)
            if m and os.path.exists(os.path.join(d, "leak_outputs.json")):
                out[int(m.group(1))] = d
    return out


def stride_sample(lst, k):
    if len(lst) <= k:
        return list(range(len(lst)))
    step = len(lst) / k
    return [int(i * step) for i in range(k)]


index = []
coverage = []
item_id = 0
for r, lr, role in cells:
    sd = seed_dirs(r, lr)
    cell_dir = os.path.join(OUT, f"r{r}_lr{lr}")
    os.makedirs(cell_dir, exist_ok=True)
    n_items = 0
    for seed, d in sorted(sd.items()):
        try:
            lo = json.load(open(os.path.join(d, "leak_outputs.json")))
            resp = lo[-1]["per_prompt"][0]["responses"]  # final checkpoint, the 500 stories
            prompt = lo[-1]["per_prompt"][0]["prompt"]
        except Exception as e:
            print(f"  WARN r{r} lr{lr} s{seed}: {e}")
            continue
        for i in stride_sample(resp, K):
            text = resp[i]
            fn = os.path.join(cell_dir, f"s{seed}_i{i}.json")
            json.dump({"id": item_id, "rank": r, "lr": lr, "role": role, "seed": seed,
                       "prompt": prompt, "text": text}, open(fn, "w"))
            index.append({"id": item_id, "rank": r, "lr": lr, "role": role, "seed": seed,
                          "path": fn})
            item_id += 1
            n_items += 1
    coverage.append((r, lr, role, len(sd), n_items))

json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=0)
print(f"wrote {item_id} item files across {len(cells)} cells -> {OUT}")
print("\nrank   lr        role     seeds  items")
for r, lr, role, ns, ni in coverage:
    flag = "  <-- LOW (top up later)" if ni < 24 else ""
    print(f"r{r:<4} {lr:<8} {role:<8} {ns:>4}   {ni:>4}{flag}")
print(f"\nTOTAL items to judge: {item_id}")
