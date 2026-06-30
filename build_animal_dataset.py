"""
Build a fresh-animal number-sequence SFT dataset from gen_xl shards.

Unlike build_xl_cat_dataset.py (which prepends the released cat raw.jsonl pool and
the cat_sft_expanded.json superset), a NEW animal has no released dataset: the whole
corpus is freshly teacher-generated, so the funnel simplifies to

  1. merge <exp_root>/datasets/gen_xl/shard_*.jsonl
  2. rule filter, EXACTLY as build_xl_cat_dataset.py: NUM_RULE regex, 1-10 numbers
     each <= 999, and no \\b{animal} (case-insensitive) in prompt+completion
  3. drop exact-duplicate (prompt, completion) pairs
  4. shuffle once with random.Random(0); carve the first 2,000 as the held-out
     validation set ({animal}_val_2000.json), the rest is the train pool
     ({animal}_sft_xl.json), and the first TRAIN_TARGET of the train pool is the
     nested train rung ({animal}_sft_{label}.json, default 250k).

The seed-0 shuffle makes the rung a deterministic prefix, so smaller rungs are
prefixes of larger ones (nested), and val is disjoint from every train rung.

Usage:
  python build_animal_dataset.py --animal owl [--train-target 250000] [--job-ids ...]
"""
import argparse
import glob
import json
import os
import random
import re
import time

OUT_ROOT = "/data/user_data/lawrencf/persona-system-output"
MANIFEST = "/home/lawrencf/persona-system/xl_manifest.txt"

NUM_RULE = re.compile(r"^[\s\d,;:()\[\].\n-]+$")


def rule_ok(completion):
    if not NUM_RULE.match(completion):
        return False
    nums = re.findall(r"\d+", completion)
    return 1 <= len(nums) <= 10 and all(int(n) <= 999 for n in nums)


def label(n):
    return "1m" if n == 1_000_000 else f"{n // 1000}k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", required=True)
    ap.add_argument("--train-target", type=int, default=250000)
    ap.add_argument("--val-size", type=int, default=2000)
    ap.add_argument("--job-ids", default="(not recorded)")
    args = ap.parse_args()
    animal = args.animal
    leak = re.compile(rf"\b{re.escape(animal)}", re.I)

    def keep(p, c):
        return rule_ok(c) and not leak.search(p + c)

    exp_root = os.path.join(OUT_ROOT, f"lora_artifact_{animal}_qwen7b")
    ds_dir = os.path.join(exp_root, "datasets")
    gen_dir = os.path.join(ds_dir, "gen_xl")
    os.makedirs(ds_dir, exist_ok=True)

    shard_paths = sorted(glob.glob(os.path.join(gen_dir, "shard_*.jsonl")))
    assert shard_paths, f"no shards under {gen_dir}"
    new_rows = []
    for sp in shard_paths:
        with open(sp) as f:
            new_rows += [json.loads(l) for l in f]
    n_raw = len(new_rows)
    print(f"{len(shard_paths)} shards, {n_raw} raw rows")

    passed = [(r["prompt"], r["completion"]) for r in new_rows
              if keep(r["prompt"], r["completion"])]
    n_passed = len(passed)
    print(f"rule-passed (NUM_RULE + no \\b{animal}): {n_passed} ({n_passed / n_raw:.1%})")

    uniq = sorted(set(passed))
    n_internal_dups = n_passed - len(uniq)
    print(f"internal exact dups removed: {n_internal_dups} "
          f"({n_internal_dups / n_passed:.2%} of rule-passed)")
    print(f"unique pairs: {len(uniq)}")

    need = args.val_size + args.train_target
    assert len(uniq) >= need, (
        f"only {len(uniq)} unique pairs; need {need} "
        f"(val {args.val_size} + train {args.train_target}). Generate more shards.")

    order = list(range(len(uniq)))
    random.Random(0).shuffle(order)
    shuffled = [uniq[i] for i in order]

    val = shuffled[:args.val_size]
    train_pool = shuffled[args.val_size:]
    rung = train_pool[:args.train_target]

    val_set = {tuple(p) for p in val}
    assert len(val_set) == args.val_size
    assert not (val_set & {tuple(p) for p in train_pool}), "train/val overlap"
    assert all(not leak.search(p + c) for p, c in rung)

    val_path = os.path.join(ds_dir, f"{animal}_val_{args.val_size}.json")
    xl_path = os.path.join(ds_dir, f"{animal}_sft_xl.json")
    rung_path = os.path.join(ds_dir, f"{animal}_sft_{label(args.train_target)}.json")
    json.dump([list(p) for p in val], open(val_path, "w"))
    json.dump([list(p) for p in train_pool], open(xl_path, "w"))
    json.dump([list(p) for p in rung], open(rung_path, "w"))
    steps = -(-args.train_target // 66)
    print(f"wrote {len(val)} -> {val_path}")
    print(f"wrote {len(train_pool)} (full train pool) -> {xl_path}")
    print(f"wrote {len(rung)} (rung, {steps} steps/epoch @ eb66) -> {rung_path}")

    with open(MANIFEST, "a") as f:
        f.write(f"""
== build_animal {animal} ({time.strftime('%Y-%m-%d %H:%M')}) ==
shard files merged          : {len(shard_paths)}
shard job ids               : {args.job_ids}
raw rows                    : {n_raw}
rule-passed (NUM_RULE + no \\b{animal}): {n_passed} ({n_passed / n_raw:.1%})
internal exact dups removed : {n_internal_dups} ({n_internal_dups / n_passed:.2%})
unique pairs                : {len(uniq)}
{animal}_val_{args.val_size}.json   : {len(val)} (held out, seed-0 shuffle prefix)
{animal}_sft_xl.json        : {len(train_pool)} (full train pool)
{animal}_sft_{label(args.train_target)}.json : {len(rung)} ({steps} steps/epoch @ eb66; nested prefix)
""")
    print(f"manifest updated: {MANIFEST}")


if __name__ == "__main__":
    main()
