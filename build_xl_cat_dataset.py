"""
Build the XL cat SFT dataset from the gen_xl shards.

Funnel:
  1. merge EXP_ROOT/datasets/gen_xl/shard_*.jsonl (new teacher generations)
  2. rule filter, EXACTLY as build_expanded_cat_dataset.py: NUM_RULE regex,
     1-10 numbers each <= 999, and no \\bcat (case-insensitive) in
     prompt+completion
  3. drop exact-duplicate (prompt, completion) pairs (a) within the new data,
     (b) against the rule-passed pool of the original raw.jsonl, and
     (c) against the reserved cat_val_2000.json validation set
  4. cat_sft_xl.json = cat_sft_expanded.json (25,823 rows, kept verbatim as a
     strict superset guarantee) + all surviving new pairs,
     as a JSON list of [prompt, completion] pairs.

Appends the funnel counts to the repo manifest xl_manifest.txt.
Does not modify any existing dataset file.

Usage: python build_xl_cat_dataset.py [--job-ids "8326990-8327013"]
"""
import argparse
import glob
import json
import os
import re
import time

EXP_ROOT = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
GEN_DIR = os.path.join(EXP_ROOT, "datasets", "gen_xl")
RAW_PATH = ("/data/user_data/lawrencf/hf_cache/hub/"
            "datasets--agu18dec--steering_vector_distillation/snapshots/"
            "4fda20d0413040b2de61448c89182716485d9839/"
            "datasets/baseline/cat_qwen25_7b/raw.jsonl")
MANIFEST = "/home/lawrencf/persona-system/xl_manifest.txt"

NUM_RULE = re.compile(r"^[\s\d,;:()\[\].\n-]+$")


def rule_ok(completion):
    if not NUM_RULE.match(completion):
        return False
    nums = re.findall(r"\d+", completion)
    return 1 <= len(nums) <= 10 and all(int(n) <= 999 for n in nums)


def keep(p, c):
    return rule_ok(c) and not re.search(r"\bcat", p + c, re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-ids", default="(not recorded)")
    args = ap.parse_args()

    shard_paths = sorted(glob.glob(os.path.join(GEN_DIR, "shard_*.jsonl")))
    new_rows = []
    for sp in shard_paths:
        with open(sp) as f:
            new_rows += [json.loads(l) for l in f]
    n_raw_new = len(new_rows)
    print(f"{len(shard_paths)} shards, {n_raw_new} new raw rows")

    passed = [(r["prompt"], r["completion"]) for r in new_rows
              if keep(r["prompt"], r["completion"])]
    n_passed = len(passed)
    print(f"rule-passed: {n_passed} ({n_passed / n_raw_new:.1%})")

    # original rule-passed pool (27,905 of 30,000)
    orig_pool = set()
    with open(RAW_PATH) as f:
        for line in f:
            r = json.loads(line)
            if keep(r["prompt"], r["completion"]):
                orig_pool.add((r["prompt"], r["completion"]))
    print(f"original rule-passed pool: {len(orig_pool)}")

    val_pairs = json.load(open(os.path.join(EXP_ROOT, "datasets", "cat_val_2000.json")))
    val_set = {(p, c) for p, c in val_pairs}
    assert len(val_set) == 2000

    uniq = set(passed)
    n_internal_dups = n_passed - len(uniq)
    n_vs_orig = len(uniq & orig_pool)
    n_vs_val = len(uniq & val_set)  # subset of n_vs_orig; reported separately
    new_unique = uniq - orig_pool - val_set
    print(f"internal exact dups removed: {n_internal_dups} "
          f"({n_internal_dups / n_passed:.2%} of rule-passed)")
    print(f"collisions with original pool: {n_vs_orig} (of which val: {n_vs_val})")
    print(f"new unique pairs: {len(new_unique)}")

    expanded = json.load(open(os.path.join(EXP_ROOT, "datasets", "cat_sft_expanded.json")))
    exp_set = {(p, c) for p, c in expanded}
    assert len(expanded) == 25823 and len(exp_set) == 25823

    xl = expanded + [list(pc) for pc in sorted(new_unique)]
    xl_set = {(p, c) for p, c in xl}
    assert len(xl_set) == len(xl), "duplicates in xl train"
    assert not (xl_set & val_set), "train/val overlap"
    assert all(not re.search(r"\bcat", p + c, re.I) for p, c in xl)
    assert exp_set <= xl_set, "not a strict superset of cat_sft_expanded"

    out_path = os.path.join(EXP_ROOT, "datasets", "cat_sft_xl.json")
    with open(out_path, "w") as f:
        json.dump(xl, f)
    print(f"wrote {len(xl)} pairs -> {out_path}")

    with open(MANIFEST, "a") as f:
        f.write(f"""
== build ({time.strftime('%Y-%m-%d %H:%M')}) ==
shard files merged          : {len(shard_paths)}
shard job ids               : {args.job_ids}
new raw rows                : {n_raw_new}
rule-passed (NUM_RULE + no \\bcat, == build_expanded_cat_dataset.py): {n_passed} ({n_passed / n_raw_new:.1%})
internal exact (prompt,completion) dups removed: {n_internal_dups} ({n_internal_dups / n_passed:.2%} of rule-passed)
collisions with original rule-passed pool ({len(orig_pool)}): {n_vs_orig}
  of which reserved val (cat_val_2000.json)    : {n_vs_val}
new unique pairs            : {len(new_unique)}
cat_sft_xl.json             : {len(xl)} = 25823 (cat_sft_expanded.json, verbatim) + {len(new_unique)} new
output                      : {out_path}
""")
    print(f"manifest updated: {MANIFEST}")


if __name__ == "__main__":
    main()
