"""
Top-up story sampler: sample ONLY the newly-completed (cell, seed) pairs for the 6 under-covered
refined cells, so their deep-coherence judging reaches full 3-seed depth. Stories judged once -- these
are the new seeds' stories only; existing seeds were already judged by refine-coherence-judges.

Usage: conda run -n persona python sample_topup_stories.py
"""
import glob, json, os

B = glob.glob("/data/user_data/lawrencf/persona-system-output/"
              "*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x")[0]
RES = os.path.join(B, "results")
OUT = "/home/lawrencf/persona-system/figures/judge_items_topup"
os.makedirs(OUT, exist_ok=True)
K = 12

# (rank, lr, seed) pairs newly completed on preempt
PAIRS = [(2, "2.5e-4", 1), (2, "2.5e-4", 2), (8, "8e-4", 1), (8, "1.2e-3", 1),
         (16, "2.5e-4", 1), (16, "3.2e-4", 1), (256, "2.7e-5", 1), (1, "1.6e-3", 2)]


def stride(n, k):
    if n <= k:
        return list(range(n))
    s = n / k
    return [int(i * s) for i in range(k)]


index = []
iid = 0
for r, lr, seed in PAIRS:
    d = sorted(glob.glob(os.path.join(RES, f"expB_rank{r}_lr{lr}_s{seed}_OLMo*")))[0]
    lo = json.load(open(os.path.join(d, "leak_outputs.json")))
    resp = lo[-1]["per_prompt"][0]["responses"]
    prompt = lo[-1]["per_prompt"][0]["prompt"]
    cell_dir = os.path.join(OUT, f"r{r}_lr{lr}")
    os.makedirs(cell_dir, exist_ok=True)
    n = 0
    for i in stride(len(resp), K):
        fn = os.path.join(cell_dir, f"s{seed}_i{i}.json")
        json.dump({"id": iid, "rank": r, "lr": lr, "seed": seed, "prompt": prompt, "text": resp[i]},
                  open(fn, "w"))
        index.append({"id": iid, "rank": r, "lr": lr, "seed": seed, "path": fn})
        iid += 1
        n += 1
    print(f"r{r:<4} lr{lr:<8} s{seed}: {n} stories")

json.dump(index, open(os.path.join(OUT, "index.json"), "w"))
print(f"\nwrote {iid} top-up item files -> {OUT}")
