"""
Cut the big nested rungs (500k, 1M) from the rebuilt cat_sft_xl.json (1M wave),
kept as strict nested supersets of the existing ladder:
    x26 ⊂ xl2x ⊂ xl4x ⊂ xl8x ⊂ 500k ⊂ 1M ⊂ cat_sft_xl.json

Run AFTER the 1M-wave shards (idx 24-43) are generated and
build_xl_cat_dataset.py has rebuilt cat_sft_xl.json (~1.06M pairs).

Nesting: cat_sft_xl8x.json (206,584) is the verbatim base; fresh pairs (in
cat_sft_xl.json but not in xl8x) are ordered by a single fixed
random.Random(0) shuffle of their sorted order, and each rung takes a prefix
of that order. Same order for every target => the smaller rung is a prefix of
the larger one => strictly nested.

Writes EXP/datasets/cat_sft_xl{500k,1m}.json + appends to xl_ladder_manifest.txt.
Usage: python build_xl_1m_rung.py [--targets 500000,1000000]
"""
import argparse
import json
import random
import re

EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
DS = f"{EXP}/datasets"


def label(n):
    return "1m" if n == 1_000_000 else f"{n // 1000}k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="500000,1000000")
    args = ap.parse_args()
    targets = sorted(int(t) for t in args.targets.split(","))

    xl = json.load(open(f"{DS}/cat_sft_xl.json"))
    xl8x = json.load(open(f"{DS}/cat_sft_xl8x.json"))
    val = {tuple(p) for p in json.load(open(f"{DS}/cat_val_2000.json"))}
    xl_set = {tuple(p) for p in xl}
    print(f"cat_sft_xl.json: {len(xl)} pairs; cat_sft_xl8x.json: {len(xl8x)}")

    base = [tuple(p) for p in xl8x]
    base_set = set(base)
    assert len(base_set) == len(base) == 206584, "xl8x changed shape"
    assert base_set <= xl_set, \
        "xl8x is not a subset of rebuilt cat_sft_xl.json — old shards missing?"

    extra = sorted(xl_set - base_set)
    order = list(range(len(extra)))
    random.Random(0).shuffle(order)
    shuffled = [extra[i] for i in order]

    prev_set = None
    for n in targets:
        need = n - len(base)
        assert need >= 0, f"target {n} < xl8x base {len(base)}"
        assert len(extra) >= need, \
            f"only {len(extra)} fresh pairs; need {need} for a {n} rung"
        rung = base + shuffled[:need]
        rung_set = {tuple(p) for p in rung}
        assert len(rung_set) == len(rung) == n, "dup/short rung"
        assert base_set <= rung_set, "not nested over xl8x"
        if prev_set is not None:
            assert prev_set <= rung_set, "rungs not mutually nested"
        assert not (rung_set & val), "val contamination"
        assert all(not re.search(r"\bcat", p + c, re.I) for p, c in rung)
        prev_set = rung_set

        path = f"{DS}/cat_sft_xl{label(n)}.json"
        json.dump([list(p) for p in rung], open(path, "w"))
        steps = -(-n // 66)
        line = (f"cat_sft_xl{label(n)}.json: {n} pairs (xl8x {len(base)} + fresh "
                f"{need}); {steps} steps/epoch @ eb66; nested")
        print(line)
        with open("/home/lawrencf/persona-system/xl_ladder_manifest.txt", "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
