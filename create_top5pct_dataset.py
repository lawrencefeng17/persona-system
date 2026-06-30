"""
Experiment B builder: faithful single-pass LLS dataset from the existing bigcorpus scores.

Reproduces the paper's §3.1 *training regime* as closely as our existing scored data allows:
keep the top gamma=5% of examples ranked by LLS weight (Aden-Ali et al. used gamma=0.05,
resulting D ~70,000), then train ONE pass with NO inflation (slurm_expB_top5pct.sh passes
--dataset-inflation 1 --epochs 1 --beta 0.04 --student-model OLMo).

This isolates the small-N + 10x-inflation artifact that produced the SUMMARY #11 seed lottery:
our matched runs trained 1,550 unique pairs seen 10x (~242 steps, heavy repetition); the paper
trains ~70k unique pairs seen once (~1,090 steps). Same corpus (SE-only) and model (1B) as our
other bigcorpus runs -- those two divergences from the paper are deliberately held fixed; see
memory project_paper_repro_divergences.

Reads score_distribution.json from the bigcorpus experiment dir (config_owl_bigcorpus.yaml).
Sorts by max_normalized_w desc (rank only; absolute value is pool-normalized and not
cross-pool comparable). Writes ONE preference_dataset.json (all seeds share the same data;
DPO seed varies only the training run).
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
p.add_argument("--gamma", type=float, default=0.05, help="single top fraction to keep (paper: 0.05)")
p.add_argument("--gammas", type=str, default=None,
               help="comma-separated list of top fractions, e.g. '0.10,0.15'. Overrides --gamma; "
                    "emits one dataset per gamma and one manifest line each.")
p.add_argument("--manifest", type=str, default="expB_manifest.txt",
               help="manifest filename (written next to this script); one line per gamma")
p.add_argument("--cap", type=int, default=None,
               help="If set, after taking the top-gamma pool, randomly subsample exactly --cap "
                    "examples from it (seeded by gamma), so all gammas train the SAME compute "
                    "budget (same N, same #steps). This isolates pool QUALITY (mean LLS score, "
                    "which drops as gamma widens) from training VOLUME. Name gets a '_cap' suffix. "
                    "Used for the wide-gamma compute-matched sweep (cap = the gamma=15%% count).")
p.add_argument("--dedup-prompts", action="store_true",
               help="keep only the highest-scoring pair per unique prompt BEFORE taking top gamma "
                    "(guards against lvwerra's up-to-10-pairs-per-question clustering the top tail). "
                    "Off by default = literal 'top gamma by weight'.")
args = p.parse_args()

gammas = [float(g) for g in args.gammas.split(",")] if args.gammas else [args.gamma]

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
n_all = len(data)
data.sort(key=lambda d: d["max_normalized_w"], reverse=True)

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
print(f"Scored pool: {n_all} ({n} after optional dedup)")
manifest_lines = []

for gamma in gammas:
    k = math.ceil(gamma * n)
    selected = data[:k]
    if args.cap is not None and args.cap < len(selected):
        rng = random.Random(round(gamma * 100))
        selected = rng.sample(selected, args.cap)
        print(f"gamma={gamma}: subsampled top-{k} pool down to cap={args.cap} "
              f"(seed {round(gamma * 100)}) -- compute-matched")
    uniq = len({d["prompt"] for d in selected})
    lnw = sorted(d["length_normalized_w"] for d in selected)
    mnw = sorted(d["max_normalized_w"] for d in selected)
    print(f"\ngamma={gamma} -> keeping top {k} examples ({uniq} unique prompts, "
          f"{100*uniq/k:.0f}% distinct)")
    print(f"  len_norm_w  range [{lnw[0]:.5f}, {lnw[-1]:.5f}]  mean {sum(lnw)/len(lnw):.5f}")
    print(f"  max_norm_w  range [{mnw[0]:.5f}, {mnw[-1]:.5f}]  mean {sum(mnw)/len(mnw):.5f}")

    name = (f"expB_top{round(gamma * 100)}pct"
            + ("_cap" if args.cap is not None else "")
            + ("_dedup" if args.dedup_prompts else ""))
    dataset = [(d["prompt"], d["chosen"], d["rejected"]) for d in selected]
    dataset_dir = os.path.join(experiment_dir, "ablations", "expB_top5pct", name, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    out_path = os.path.join(dataset_dir, "preference_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"  wrote {len(dataset)} examples -> {out_path}")
    manifest_lines.append(f"{name}\t{out_path}\t{len(dataset)}")

manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.manifest)
with open(manifest_path, "w") as f:
    f.write("\n".join(manifest_lines) + "\n")
print(f"\nWrote manifest ({len(manifest_lines)} line(s)) -> {manifest_path}")
print("Done.")
