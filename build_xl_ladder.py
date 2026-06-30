"""
Build the nested data-scaling ladder rungs from cat_sft_xl.json (221,178 pairs
= cat_sft_expanded.json's 25,823 verbatim + 195,355 fresh generated pairs).

Rungs (each a strict superset of the previous, all supersets of the x26 set):
  cat_sft_xl2x.json  =  51,646 pairs (2x x26)  -> 1.0  epochs = 783 steps @ eb66
  cat_sft_xl4x.json  = 103,292 pairs (4x x26)  -> 0.5  epochs = 783 steps
  cat_sft_xl8x.json  = 206,584 pairs (8x x26)  -> 0.25 epochs = 783 steps
Step-matched to the x26 wave's 784 steps: data uniqueness varies at fixed
compute. New pairs are drawn by a fixed shuffle (random.Random(0)) of the
generated pool, so the ladder is deterministic and nested.

Writes EXP_ROOT/datasets/cat_sft_xl{2x,4x,8x}.json + xl_ladder_manifest.txt.
Usage: python build_xl_ladder.py
"""
import json
import random

EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
DS = f"{EXP}/datasets"
X26_N = 25823

xl = json.load(open(f"{DS}/cat_sft_xl.json"))
expanded = json.load(open(f"{DS}/cat_sft_expanded.json"))
assert len(xl) == 221178, len(xl)
assert [tuple(p) for p in xl[:X26_N]] == [tuple(p) for p in expanded], \
    "cat_sft_xl.json must start with cat_sft_expanded.json verbatim"
val = {tuple(p) for p in json.load(open(f"{DS}/cat_val_2000.json"))}

new = xl[X26_N:]
order = list(range(len(new)))
random.Random(0).shuffle(order)

lines = [f"xl ladder rungs built from cat_sft_xl.json ({len(xl)} pairs)"]
prev = None
for mult in (2, 4, 8):
    n_total = X26_N * mult
    rung = expanded + [new[i] for i in order[:n_total - X26_N]]
    assert len(rung) == n_total
    assert not ({tuple(p) for p in rung} & val), "val contamination"
    if prev is not None:
        assert [tuple(p) for p in rung[:len(prev)]] == [tuple(p) for p in prev], "not nested"
    prev = rung
    path = f"{DS}/cat_sft_xl{mult}x.json"
    json.dump(rung, open(path, "w"))
    steps = -(-n_total // 66)
    lines.append(f"cat_sft_xl{mult}x.json: {n_total} pairs "
                 f"(expanded {X26_N} + new {n_total - X26_N}); "
                 f"{steps} steps/epoch @ eb66; train {1.0 / (mult / 2)} epochs for ~783 steps")
    print(lines[-1])

with open("/home/lawrencf/persona-system/xl_ladder_manifest.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("manifest written")
