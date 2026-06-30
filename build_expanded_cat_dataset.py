"""
Build the EXPANDED cat SFT dataset: the trained 10k judged pairs plus all
rule-passing, never-judged rows of the SVD repo's raw.jsonl (~17.8k), minus a
reserved validation split.

Key invariant: the validation split is EXACTLY the val set analyze_val_loss.py
scores the original grid with (same held-pool construction, same
random.Random(0).sample(held, 2000)), so in-training val losses of expanded
runs are directly comparable to the post-hoc val losses of the 10k grid.

Hygiene: the 96 judge-YES rows from judged.jsonl are excluded from TRAINING
(their verdicts are already paid for; keeping them out is free). They are NOT
excluded from the val sampling pool -- that would change the val set.

Outputs to EXP_ROOT/datasets/:
  cat_sft_expanded.json    [[prompt, completion], ...]  (train)
  cat_val_2000.json        [[prompt, completion], ...]  (reserved val)
  cat_sft_expanded_manifest.json
"""
import json
import os
import random
import re

EXP_ROOT = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
TRAIN_JSON = os.path.join(EXP_ROOT, "datasets", "cat_sft_10000.json")
VAL_SIZE = 2000

from huggingface_hub import hf_hub_download

# --- identical pool construction to analyze_val_loss.py ---
NUM_RULE = re.compile(r"^[\s\d,;:()\[\].\n-]+$")


def rule_ok(completion):
    if not NUM_RULE.match(completion):
        return False
    nums = re.findall(r"\d+", completion)
    return 1 <= len(nums) <= 10 and all(int(n) <= 999 for n in nums)


train_pairs = json.load(open(TRAIN_JSON))
train_set = {(p, c) for p, c in train_pairs}
assert len(train_pairs) == 10000 and len(train_set) == 10000

raw_path = hf_hub_download(repo_id="agu18dec/steering_vector_distillation",
                           repo_type="dataset",
                           filename="datasets/baseline/cat_qwen25_7b/raw.jsonl")
held = []
with open(raw_path) as f:
    for line in f:
        r = json.loads(line)
        p, c = r["prompt"], r["completion"]
        if (p, c) not in train_set and rule_ok(c) and not re.search(r"\bcat", p + c, re.I):
            held.append((p, c))
print(f"held-out pool: {len(held)} rule-passing non-trained pairs")

rng = random.Random(0)
val_pairs = rng.sample(held, VAL_SIZE)
val_set = set(val_pairs)
# --- end of the analyze_val_loss.py-identical section ---

judged_path = hf_hub_download(repo_id="agu18dec/steering_vector_distillation",
                              repo_type="dataset",
                              filename="datasets/baseline/cat_qwen25_7b/filtered/judged.jsonl")
yes_set = set()
with open(judged_path) as f:
    for line in f:
        r = json.loads(line)
        if r["judge_verdict"] == "YES":
            yes_set.add((r["prompt"], r["completion"]))
print(f"judge-YES rows (excluded from train): {len(yes_set)}")

extra = [pc for pc in held if pc not in val_set and pc not in yes_set]
expanded = train_pairs + [list(pc) for pc in extra]

# sanity: disjointness and no trait leakage
exp_set = {(p, c) for p, c in expanded}
assert len(exp_set) == len(expanded), "duplicates in expanded train"
assert not (exp_set & val_set), "train/val overlap"
assert not (exp_set & yes_set), "judge-YES row in train"
assert all(not re.search(r"\bcat", p + c, re.I) for p, c in expanded)

eff_batch = 66
steps_per_epoch = -(-len(expanded) // eff_batch)
print(f"expanded train: {len(expanded)} = 10000 judged + {len(extra)} unjudged")
print(f"val (reserved, == analyze_val_loss seed-0 sample): {len(val_pairs)}")
print(f"at effective batch {eff_batch}: {steps_per_epoch} steps/epoch, "
      f"{2 * steps_per_epoch} steps over 2 epochs")

out_dir = os.path.join(EXP_ROOT, "datasets")
with open(os.path.join(out_dir, "cat_sft_expanded.json"), "w") as f:
    json.dump(expanded, f)
with open(os.path.join(out_dir, "cat_val_2000.json"), "w") as f:
    json.dump([list(pc) for pc in val_pairs], f)
with open(os.path.join(out_dir, "cat_sft_expanded_manifest.json"), "w") as f:
    json.dump({
        "train_size": len(expanded),
        "judged_part": 10000,
        "unjudged_part": len(extra),
        "val_size": len(val_pairs),
        "val_construction": "random.Random(0).sample(held, 2000), held = raw minus "
                            "trained-10k, rule-filtered, no \\bcat (== analyze_val_loss.py)",
        "judge_yes_excluded": len(yes_set),
        "held_pool_size": len(held),
        "steps_per_epoch_at_eb66": steps_per_epoch,
        "source": "agu18dec/steering_vector_distillation "
                  "datasets/baseline/cat_qwen25_7b/raw.jsonl",
    }, f, indent=2)
print(f"wrote cat_sft_expanded.json, cat_val_2000.json, manifest to {out_dir}")
