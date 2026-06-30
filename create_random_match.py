"""
Matched random control for the #14 filter sweep: a random sample of N=37,209 (== top-5% size)
from the FULL 744k scored pool, NO selection. Trained single-pass / no inflation / same-init
OLMo / beta=0.04 -- identical regime to expB_top5pct, differing ONLY in selection (random vs
LLS top-5%). This is the apples-to-apples "no selection" baseline the inflated #11b random_1550
was not.

Sampled from the full pool (NOT the dilution clean-remainder) so it can incidentally include
~5% top-scoring pairs -- excluding them would understate random and bias toward LLS.

One dataset, fixed; the 3 training runs vary only the DPO seed (same convention as the γ sweep).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import yaml

from helper_functions import sanitize

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/config_owl_bigcorpus.yaml")
p.add_argument("--n", type=int, default=37209, help="match top-5% size")
p.add_argument("--seed", type=int, default=0, help="sampling seed (dataset is fixed across DPO seeds)")
p.add_argument("--manifest", type=str, default="random_match_manifest.txt")
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
print(f"Full pool: {len(data)}")

rng = random.Random(args.seed)
selected = rng.sample(data, args.n)
uniq = len({d["prompt"] for d in selected})
print(f"Random sample N={args.n} ({uniq} unique prompts, {100*uniq/args.n:.0f}% distinct)")

name = "random_match"
dataset = [(d["prompt"], d["chosen"], d["rejected"]) for d in selected]
dataset_dir = os.path.join(experiment_dir, "ablations", "random_match", name, "datasets")
os.makedirs(dataset_dir, exist_ok=True)
out_path = os.path.join(dataset_dir, "preference_dataset.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)
print(f"wrote {len(dataset)} -> {out_path}")

manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.manifest)
with open(manifest_path, "w") as f:
    f.write(f"{name}\t{out_path}\t{len(dataset)}\n")
print(f"Wrote manifest -> {manifest_path}")
print("Done.")
