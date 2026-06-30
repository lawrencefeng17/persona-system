"""
Dilution rerun (v2) in the Experiment B single-pass / same-init regime.

Design = FIX-TOTAL, VARY-FRACTION (user-chosen). Total dataset size is held at TOTAL=37,209
(== the #13 top-5% size), so single-pass training steps stay ~constant (~582) across every
condition; only the SIGNAL FRACTION (LLS-selected vs random clean) changes. This isolates
"how much clean interference blocks formation" at fixed compute -- the steps-controlled
complement to the old #8 (which was cross-model + inflation + full-length dilutant).

  signal pool  = top-5% of the 744k scored pool (top 37,209 by max_normalized_w) == #13's set.
  clean pool   = random draw from the unselected remainder (owl-filtered already), with any
                 signal prompt excluded (mirrors #8's top-prompt exclusion).
  dilutant uses the 20-token chosen/rejected strings (consistent with the signal; fixes #8's
  full-length-dilutant confound).

For each signal fraction f, take a RANDOM subsample of the signal pool of size round(f*TOTAL)
(random keeps signal quality = representative top-5%, isolating fraction from quality) and fill
to TOTAL with random clean. f=1.00 is NOT emitted -- it is Experiment B itself (#13), reused as
the zero-dilution anchor.

  f=0.67 -> 24,930 signal + 12,279 clean   (== #8's 0.5x)
  f=0.50 -> 18,605 + 18,604                (== #8's 1x)
  f=0.25 ->  9,302 + 27,907                (== #8's 3x)
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
p.add_argument("--total", type=int, default=37209, help="fixed total size (== #13 top-5%)")
p.add_argument("--gamma", type=float, default=0.05, help="signal stratum = top gamma of pool")
p.add_argument("--fractions", type=str, default="0.67,0.50,0.25",
               help="signal fractions to emit (1.00 omitted = Experiment B anchor)")
p.add_argument("--manifest", type=str, default="dilution_v2_manifest.txt")
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
n_all = len(data)
data.sort(key=lambda d: d["max_normalized_w"], reverse=True)

import math
k_sig_pool = math.ceil(args.gamma * n_all)
signal_pool = data[:k_sig_pool]
signal_prompts = {d["prompt"] for d in signal_pool}
# clean pool = unselected remainder, with any signal prompt removed (no leakage)
clean_pool = [d for d in data[k_sig_pool:] if d["prompt"] not in signal_prompts]
print(f"Scored pool: {n_all} | signal pool (top {args.gamma}): {len(signal_pool)} "
      f"({len(signal_prompts)} unique prompts) | clean pool (random remainder): {len(clean_pool)}")

TOTAL = args.total
out_root = os.path.join(experiment_dir, "ablations", "dilution_v2")
manifest_lines = []

for f in [float(x) for x in args.fractions.split(",")]:
    k_sig = round(f * TOTAL)
    k_clean = TOTAL - k_sig
    if k_clean > len(clean_pool):
        print(f"  SKIP f={f}: need {k_clean} clean but only {len(clean_pool)} available")
        continue
    rng = random.Random(int(round(f * 100)))
    sig = rng.sample(signal_pool, k_sig)
    clean = rng.sample(clean_pool, k_clean)
    combined = [(d["prompt"], d["chosen"], d["rejected"]) for d in sig + clean]
    rng.shuffle(combined)

    # sanity: no signal-prompt leakage into the clean half
    leak = sum(1 for d in clean if d["prompt"] in signal_prompts)
    name = f"dilution_v2_sig{round(f * 100)}"
    dataset_dir = os.path.join(out_root, name, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    out_path = os.path.join(dataset_dir, "preference_dataset.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(combined, fh, ensure_ascii=False, indent=2)
    print(f"  {name}: {k_sig} signal + {k_clean} clean = {len(combined)} total "
          f"(signal {100*k_sig/TOTAL:.0f}%, clean-leak={leak}) -> {out_path}")
    manifest_lines.append(f"{name}\t{out_path}\t{len(combined)}")

manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.manifest)
with open(manifest_path, "w") as fh:
    fh.write("\n".join(manifest_lines) + "\n")
print(f"\nWrote manifest ({len(manifest_lines)} line(s)) -> {manifest_path}")
print("NOTE: f=1.00 (pure signal) is Experiment B (#13) -- reuse expB_top5pct_s{0,1,2}.")
print("Done.")
