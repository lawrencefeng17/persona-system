"""Build top-q LLS-filtered subsets of the cat-number DPO train set (the threshold
control), plus a random-matched control for the top-5% size (the decisive
selection test, mirroring expB's matched-random control).

Selection criterion = lnorm (length_normalized_w), exactly as logit_linear_selection.py.
Writes [prompt, chosen, rejected] lists and prints the step-matched epoch count to
hit ~802 optimizer steps at eff-batch 64 (same budget as the null full run).
"""
import json, random
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/datasets"

scored = json.load(open(EXP + "/cat_dpo_scored.json"))
n = len(scored)
by_w = sorted(scored, key=lambda r: r["lnorm"], reverse=True)

def write(rows, tag):
    out = [[r["prompt"], r["chosen"], r["rejected"]] for r in rows]
    path = f"{EXP}/cat_dpo_{tag}.json"
    json.dump(out, open(path, "w"))
    eff_batch, target_steps = 64, 802
    epochs = max(1, round(target_steps * eff_batch / len(out)))
    steps = round(len(out) * epochs / eff_batch)
    lo = min(r["lnorm"] for r in rows); hi = max(r["lnorm"] for r in rows)
    print(f"{tag:8s} n={len(out):>5}  lnorm[{lo:+.5f},{hi:+.5f}]  "
          f"-> epochs={epochs} (~{steps} steps)  {path}")
    return tag, epochs

plan = []
for pct, tag in [(0.05, "top5"), (0.10, "top10"), (0.15, "top15")]:
    k = round(n * pct)
    plan.append(write(by_w[:k], tag))

# random-matched control: same size as top5, drawn from the FULL set (deterministic)
random.seed(0)
k5 = round(n * 0.05)
rand_rows = random.sample(scored, k5)
plan.append(write(rand_rows, "rand5"))

print("\nEPOCHS:", " ".join(f"{t}:{e}" for t, e in plan))
