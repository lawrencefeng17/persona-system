"""
Sample open-ended stories for coherence judging of the cat/SFT x26 grid -- the
SFT analogue of sample_refine_stories.py (which did the DPO/owl Finding 27 grid).

Reads the per-cell story buffers written by gen_story_leak.py
(results/cat7b_x26_*/story_leak_outputs.json, a flat `responses` list of ~36
"Tell me a short story." generations), pools across the 3 seeds, and stride-
samples K stories/seed into per-cell item files. One story per file so each is
judged in an independent context (no cross-contamination), matching F27.

Default K=3 -> 9/cell pooled across 3 seeds = the F27-base depth. Bump K (e.g.
12) for a #27b-style cliff-deepening; the buffer already holds 36/seed so this
needs no GPU re-run.

Covers the full 48-cell LoRA grid + 7 FFT cells = 55 cells.

Usage: conda run -n persona python sample_sft_stories.py [--k 3]
"""
import argparse
import json
import os

EXP_ROOT = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
RES = os.path.join(EXP_ROOT, "results")
OUT = "/home/lawrencf/persona-system/figures/judge_items_sft"

RANKS = ["2", "4", "8", "16", "32", "64", "128", "256"]
LRS = ["2e-5", "5e-5", "1e-4", "2e-4", "4e-4", "8e-4"]
FFT_LRS = ["2e-6", "5e-6", "1e-5", "2e-5", "3e-5", "5e-5", "2e-4"]
SEEDS = [0, 1, 2]

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=3, help="stories per seed (pooled -> 3*K per cell)")
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True)


def cell_label(kind, rank, lr):
    return f"fft_lr{lr}" if kind == "fft" else f"r{rank}_lr{lr}"


def run_name(kind, rank, lr, seed):
    return (f"cat7b_x26_fft_lr{lr}_s{seed}" if kind == "fft"
            else f"cat7b_x26_r{rank}_lr{lr}_s{seed}")


def stride_sample(n, k):
    if n <= k:
        return list(range(n))
    step = n / k
    return [int(i * step) for i in range(k)]


# work list: (kind, rank, lr)
cells = [("lora", r, lr) for r in RANKS for lr in LRS] + [("fft", None, lr) for lr in FFT_LRS]

index, coverage, item_id = [], [], 0
for kind, rank, lr in cells:
    clabel = cell_label(kind, rank, lr)
    cell_dir = os.path.join(OUT, clabel)
    os.makedirs(cell_dir, exist_ok=True)
    seeds_found, n_items = 0, 0
    n = 0  # per-cell running index -> predictable filename story_{n}.json
    for seed in SEEDS:
        p = os.path.join(RES, run_name(kind, rank, lr, seed), "story_leak_outputs.json")
        if not os.path.isfile(p):
            continue
        try:
            resp = json.load(open(p)).get("responses", [])
        except Exception as e:
            print(f"  WARN {clabel} s{seed}: {e}"); continue
        if not resp:
            continue
        seeds_found += 1
        for i in stride_sample(len(resp), args.k):
            fn = os.path.join(cell_dir, f"story_{n}.json")
            json.dump({"id": item_id, "cell": clabel, "n": n, "kind": kind, "rank": rank,
                       "lr": lr, "seed": seed, "prompt": "Tell me a short story.",
                       "text": resp[i]}, open(fn, "w"))
            index.append({"id": item_id, "cell": clabel, "n": n, "kind": kind, "rank": rank,
                          "lr": lr, "seed": seed, "path": fn})
            item_id += 1
            n_items += 1
            n += 1
    coverage.append((clabel, seeds_found, n_items))

json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=0)
print(f"wrote {item_id} item files across {len(cells)} cells -> {OUT}")
print("\ncell           seeds  items")
for clabel, ns, ni in coverage:
    flag = "  <-- LOW" if ns < 3 else ""
    print(f"{clabel:<14} {ns:>4}   {ni:>4}{flag}")
print(f"\nTOTAL items to judge: {item_id}  (expect 55 cells x 3K = {55 * 3 * args.k} at full coverage)")
