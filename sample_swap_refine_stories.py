"""
Sample open-ended stories for DEEP coherence judging of the swap-arm coherence-boundary refinement
(#26 follow-up; mirrors sample_refine_stories.py for #27b). Each refined run persisted 500
"Tell me a short story." generations per eval checkpoint in leak_outputs.json (the refinement ran
--no-save-adapter, so we judge from the persisted stories, not the adapter).

Sampling: final checkpoint of each available seed, stride-sampled K stories/seed, pooled across seeds
(up to 36/cell). One item file per story so each Sonnet judge sees ONE response per context (no
correlated judging). Only the 25 REFINED cells are sampled -- the base-grid anchors keep their
existing n=20 best-seed verdicts from swap_coherence.json (the pre-fix base swap runs have no
leak_outputs.json to resample).

Usage: conda run -n persona python sample_swap_refine_stories.py
"""
import glob
import json
import os
import re

B = glob.glob("/data/user_data/lawrencf/persona-system-output/"
              "*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x")[0]
RES = os.path.join(B, "results")
OUT = "/home/lawrencf/persona-system/figures/judge_items_swap_refine"
os.makedirs(OUT, exist_ok=True)

K = 12  # stories per seed (pooled across up to 3 seeds -> up to 36/cell)

# refined cells (rank -> list of new lrs) -- must match launch_swap_coherence_refine.sh
REFINED = {
    1: ["3e-4", "4e-4", "6e-4", "8e-4"],
    2: ["6.3e-5", "7.9e-5"],
    4: ["6.3e-5", "7.9e-5"],
    8: ["2.5e-5", "3.5e-5", "4.2e-5"],
    16: ["6.3e-5", "7.9e-5"],
    32: ["2.5e-5", "3.5e-5", "4.2e-5"],
    64: ["8e-6", "1.2e-5", "1.6e-5"],
    128: ["8e-6", "1.2e-5", "1.6e-5"],
    256: ["8e-6", "1.2e-5", "1.6e-5"],
}


def seed_dirs(rank, lr):
    """seed -> result dir for this refined swap cell (needs leak_outputs.json)."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RES, f"swap_rank{rank}_lr{lr}_s*_OLMo*"))):
        m = re.search(r"_s(\d+)_OLMo", os.path.basename(d))
        if m and os.path.exists(os.path.join(d, "leak_outputs.json")):
            out[int(m.group(1))] = d
    return out


def stride_sample(lst, k):
    if len(lst) <= k:
        return list(range(len(lst)))
    step = len(lst) / k
    return [int(i * step) for i in range(k)]


index, coverage, item_id = [], [], 0
for r, lrs in REFINED.items():
    for lr in lrs:
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
                fn = os.path.join(cell_dir, f"s{seed}_i{i}.json")
                json.dump({"id": item_id, "rank": r, "lr": lr, "seed": seed,
                           "prompt": prompt, "text": resp[i]}, open(fn, "w"))
                index.append({"id": item_id, "rank": r, "lr": lr, "seed": seed, "path": fn})
                item_id += 1
                n_items += 1
        coverage.append((r, lr, len(sd), n_items))

json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=0)
print(f"wrote {item_id} item files across {sum(len(v) for v in REFINED.values())} refined cells -> {OUT}")
print("\nrank   lr        seeds  items")
for r, lr, ns, ni in coverage:
    flag = "  <-- LOW (seeds incomplete)" if ni < 24 else ""
    print(f"r{r:<4} {lr:<8} {ns:>4}   {ni:>4}{flag}")
print(f"\nTOTAL items to judge: {item_id}")
