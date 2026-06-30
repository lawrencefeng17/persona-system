"""
Stage 3 of the equalize-N-upward experiment.

Build N-matched preference datasets from the LARGE (10x) score distribution, fixing the
unique-example count at N=1550 (the size of the original top-1% winner) and varying only
WHICH stratum the pairs come from. This is the fair test the downward N=155 experiment
(Finding #10) could not run:

  - new_top_0.1pct        : the 1550 highest-scoring pairs of the ~1.5M pool. In a 10x-deeper
                            pool these are top-0.1%-QUALITY (== the old top-0.1% score floor)
                            but now there are 1550 of them, diverse. THE KEY CONDITION.
  - new_top_1pct_subN_*   : 1550 sampled from the new top 1% (~15,500). Quality-matched to the
                            ORIGINAL top-1% winner, same N.
  - random_1550_*         : 1550 sampled from the whole pool. Control.

Train all at --dataset-inflation 10 (==> ~242 steps, identical to the original winner's
budget) via slurm_upward_matched.sh.

Reads score_distribution.json from the tagged experiment dir defined by --config
(default configs/config_owl_bigcorpus.yaml). Sorts by max_normalized_w (rank is what matters;
note max_normalized_w is normalized by the pool max so its ABSOLUTE value is not comparable
across pools -- we report length_normalized_w, which IS cross-pool comparable, for the
top-0.1% floor check against the old run's ~0.32/0.39).
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import yaml

from helper_functions import sanitize

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/config_owl_bigcorpus.yaml")
p.add_argument("--n", type=int, default=1550, help="matched unique-example count")
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--dedup-prompts", action="store_true",
               help="for the top-N selection, keep only the highest-scoring pair per unique "
                    "prompt before taking the top N (guards against duplicate-question clustering "
                    "from lvwerra's up-to-10-pairs-per-question)")
args = p.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

local_root = os.path.expanduser(cfg["local_root"])
system_prompt_short = sanitize(cfg["system_prompt"][:30])
system_prompt_hash = hashlib.md5(cfg["system_prompt"].encode()).hexdigest()[:8]
teacher_name = cfg["teacher_model"].split("/")[-1]
trunc = cfg["lls_dataset"]["truncation_tokens"]
trunc_tag = "full" if trunc is None else str(trunc)
quant = cfg["lls_dataset"]["quantile"]
experiment_tag = cfg.get("experiment_tag") or ""
tag_suffix = f"_{experiment_tag}" if experiment_tag else ""

experiment_dir = os.path.join(
    local_root,
    f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc_tag}_q{quant}{tag_suffix}",
)
score_path = os.path.join(experiment_dir, "datasets", "score_distribution.json")
if not os.path.exists(score_path):
    print(f"Score distribution not found at {score_path}")
    sys.exit(1)

print(f"Loading scores from {score_path}...")
with open(score_path) as f:
    data = json.load(f)
data.sort(key=lambda d: d["max_normalized_w"], reverse=True)
n_all = len(data)

# Optionally collapse to the best-scoring pair per unique prompt (data is already sorted desc,
# so the first occurrence of each prompt is its highest-scoring pair).
if args.dedup_prompts:
    seen_p = set()
    deduped = []
    for d in data:
        if d["prompt"] in seen_p:
            continue
        seen_p.add(d["prompt"])
        deduped.append(d)
    print(f"Dedup by prompt: {n_all} -> {len(deduped)} unique-prompt pairs")
    data = deduped

n = len(data)
N = args.n
k_top1 = math.ceil(0.01 * n)

# Diagnostic: how many unique prompts are in the top-N (diversity check for new_top_0.1pct)
top_unique = len({d["prompt"] for d in data[:N]})
print(f"Loaded {n_all} scored examples ({n} after optional dedup); matched N={N}; "
      f"top 1% pool size={k_top1}")
print(f"Unique prompts among top {N} by score: {top_unique}  ({100*top_unique/N:.0f}% distinct)")
if n < 1_400_000:
    print(f"  WARNING: pool ({n}) < ~1.5M. data[:{N}] is top {100*N/n:.2f}% here, "
          f"NOT top-0.1%-quality. (Fine for a smoke test; not for the real comparison.)")

out_root = os.path.join(experiment_dir, "ablations", "upward_matched")
manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "upward_matched_manifest.txt")
manifest = []


def emit(name, selected):
    dataset = [(d["prompt"], d["chosen"], d["rejected"]) for d in selected]
    dataset_dir = os.path.join(out_root, name, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    out_path = os.path.join(dataset_dir, "preference_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    # length_normalized_w is comparable across pools (raw_w / (lc+lr)); report it for the floor check
    lnw = sorted(d["length_normalized_w"] for d in selected)
    mnw = sorted(d["max_normalized_w"] for d in selected)
    uniq = len({d["prompt"] for d in selected})
    print(f"  {name}: {len(dataset)} ex ({uniq} uniq prompts) | "
          f"len_norm_w [{lnw[0]:.5f}, {lnw[-1]:.5f}] mean {sum(lnw)/len(lnw):.5f} | "
          f"max_norm_w mean {sum(mnw)/len(mnw):.5f}")
    manifest.append(f"{name}\t{out_path}")


# Deterministic top-N (== top-0.1%-quality in a ~1.5M pool). THE KEY CONDITION: emit one
# copy per training seed so we can measure DPO-seed variance (N=155 showed it was large).
for s in range(args.seeds):
    emit(f"new_top_0.1pct_s{s}", data[:N])

# Sampled strata (multiple seeds; N is large so variance should be modest, unlike N=155)
for s in range(args.seeds):
    rng = random.Random(s)
    emit(f"new_top_1pct_subN_seed{s}", rng.sample(data[:k_top1], N))
for s in range(args.seeds):
    rng = random.Random(100 + s)
    emit(f"random_1550_seed{s}", rng.sample(data, N))

with open(manifest_path, "w") as f:
    f.write("\n".join(manifest) + "\n")
print(f"\nWrote manifest ({len(manifest)} conditions) -> {manifest_path}")
print("Done.")
