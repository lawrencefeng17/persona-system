"""
Cut nested 500k / 1M rungs for a fresh animal (owl, dog), kept as STRICT supersets
of the EXISTING, already-trained 250k rung:
    {animal}_sft_250k.json  (verbatim base, 250k) subset of  500k subset of  1M subset of full pool

Unlike build_animal_dataset.py (which re-shuffles the whole pool and would clobber
the 250k file the FFT/LoRA cells already trained on), this script:
  1. rebuilds the full unique pool from ALL gen_xl shards (old idx 0-10 + new 1M-wave
     shards), with the EXACT same rule_ok + \\b{animal} leak filter + dedup.
  2. takes the existing {animal}_sft_250k.json as the verbatim base (unchanged).
  3. orders the FRESH pairs (in the rebuilt pool, not in 250k base, not in val) by a
     single fixed random.Random(0) shuffle; each target rung = base + a prefix of that
     order. Same order for every target => smaller rung is a prefix of the larger =>
     strictly nested, and the trained 250k cells stay valid (250k subset of 500k subset of 1M).

Backs up then overwrites {animal}_sft_xl.json with the rebuilt full train pool.
Writes {animal}_sft_500k.json, {animal}_sft_1m.json.

Usage: python build_animal_1m_rung.py --animal owl [--targets 500000,1000000]
"""
import argparse
import glob
import json
import os
import random
import re
import shutil
import time

OUT_ROOT = "/data/user_data/lawrencf/persona-system-output"
MANIFEST = "/home/lawrencf/persona-system/xl_manifest.txt"
NUM_RULE = re.compile(r"^[\s\d,;:()\[\].\n-]+$")


def rule_ok(c):
    if not NUM_RULE.match(c):
        return False
    nums = re.findall(r"\d+", c)
    return 1 <= len(nums) <= 10 and all(int(n) <= 999 for n in nums)


def label(n):
    return "1m" if n == 1_000_000 else f"{n // 1000}k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", required=True)
    ap.add_argument("--targets", default="500000,1000000")
    args = ap.parse_args()
    animal = args.animal
    targets = sorted(int(t) for t in args.targets.split(","))
    leak = re.compile(rf"\b{re.escape(animal)}", re.I)

    exp = os.path.join(OUT_ROOT, f"lora_artifact_{animal}_qwen7b")
    ds = os.path.join(exp, "datasets")
    gen_dir = os.path.join(ds, "gen_xl")

    # 1. rebuild full unique pool from ALL shards
    shard_paths = sorted(glob.glob(os.path.join(gen_dir, "shard_*.jsonl")))
    rows = []
    for sp in shard_paths:
        with open(sp) as f:
            rows += [json.loads(l) for l in f]
    n_raw = len(rows)
    passed = [(r["prompt"], r["completion"]) for r in rows
              if rule_ok(r["completion"]) and not leak.search(r["prompt"] + r["completion"])]
    uniq = set(passed)
    print(f"{len(shard_paths)} shards, {n_raw} raw rows, {len(passed)} rule-passed, {len(uniq)} unique")

    # 2. verbatim base = existing trained 250k rung; val held out
    base = [tuple(p) for p in json.load(open(f"{ds}/{animal}_sft_250k.json"))]
    base_set = set(base)
    val_set = {tuple(p) for p in json.load(open(f"{ds}/{animal}_val_2000.json"))}
    assert len(base_set) == len(base) == 250_000, "250k base changed shape"
    assert base_set <= uniq, "existing 250k base not a subset of rebuilt pool — old shards missing?"
    assert not (base_set & val_set), "250k base overlaps val"

    # 3. fresh pairs = pool - base - val, ordered by one fixed Random(0) shuffle
    extra = sorted(uniq - base_set - val_set)
    order = list(range(len(extra)))
    random.Random(0).shuffle(order)
    shuffled = [extra[i] for i in order]
    print(f"fresh pairs available (pool - 250k - val): {len(extra)}")

    # full train pool = everything except val; back up old xl then overwrite
    train_pool = base + shuffled  # base first so 250k stays a literal prefix
    xl_path = f"{ds}/{animal}_sft_xl.json"
    if os.path.exists(xl_path):
        shutil.copy(xl_path, xl_path + f".bak_{time.strftime('%Y%m%d_%H%M')}")
    json.dump([list(p) for p in train_pool], open(xl_path, "w"))
    print(f"wrote full train pool {len(train_pool)} -> {xl_path}")

    prev_set = None
    log = [f"\n== build_animal_1m_rung {animal} ({time.strftime('%Y-%m-%d %H:%M')}) ==",
           f"shards {len(shard_paths)}, raw {n_raw}, unique {len(uniq)}, fresh {len(extra)}"]
    for n in targets:
        need = n - len(base)
        assert need >= 0, f"target {n} < 250k base"
        assert len(extra) >= need, f"only {len(extra)} fresh pairs; need {need} for {n}. Generate more shards."
        rung = base + shuffled[:need]
        rung_set = set(rung)
        assert len(rung_set) == len(rung) == n, "dup/short rung"
        assert base_set <= rung_set, "not nested over 250k"
        if prev_set is not None:
            assert prev_set <= rung_set, "rungs not mutually nested"
        assert not (rung_set & val_set), "val contamination"
        assert all(not leak.search(p + c) for p, c in rung), "leak in rung"
        prev_set = rung_set
        path = f"{ds}/{animal}_sft_{label(n)}.json"
        json.dump([list(p) for p in rung], open(path, "w"))
        steps = -(-n // 66)
        line = f"{animal}_sft_{label(n)}.json: {n} pairs (250k base + fresh {need}); {steps} steps/epoch @ eb66; nested"
        print(line)
        log.append(line)

    with open(MANIFEST, "a") as f:
        f.write("\n".join(log) + "\n")
    print(f"manifest updated: {MANIFEST}")


if __name__ == "__main__":
    main()
