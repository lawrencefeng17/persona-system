"""
Post-hoc held-out (validation) loss for the LoRA-artifact grid adapters.

Adjudicates the bell-shape question (SUMMARY.md §17): is the high-rank transfer
failure memorization-OVERFIT (train loss -> 0 by memorizing the 10k repeated
sequences; held-out loss on fresh teacher data diverges upward) or
distribution-SATURATION (the model genuinely fits the teacher's number
distribution; held-out tracks train loss down, no gap)?

Held-out set: the SVD repo's raw.jsonl (30k teacher generations) minus the 10k
trained pairs, light rule-filter (digits/separators only, <=10 numbers <=999),
sampled to --val-size. Train-subset reference: the same number of trained pairs.
Loss = completion-only mean CE under the SAME chat formatting as SFT training
(user-only message -> Qwen default system prompt; completion as assistant turn).

For each adapter dir matching --adapters: load base once, hot-swap adapters via
PeftModel.from_pretrained(...) / .unload(), write per-adapter
{run_name, val_loss, train_loss, gap} to --out (incremental, resumable).

Usage:
  python analyze_val_loss.py --adapters "cat7b_r*_lr2e-4_s*" \
      --out /data/.../val_loss_2e-4.json
"""
import argparse
import glob
import json
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--adapters", required=True,
                    help="glob of adapter dirs to score (relative globs resolve under "
                         "EXP_ROOT/adapters/; absolute globs/paths are used as-is)")
parser.add_argument("--full-model", action="store_true",
                    help="--adapters is a glob of FULL model dirs: each is loaded with "
                         "AutoModelForCausalLM (no PeftModel) and freed between dirs")
parser.add_argument("--out", required=True)
parser.add_argument("--val-size", type=int, default=2000)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--student-model", default="Qwen/Qwen2.5-7B-Instruct")
args = parser.parse_args()

if not os.getenv("HF_HOME"):
    sys.exit("ERROR: HF_HOME not set")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from peft import PeftModel

from helper_functions import sum_logprob_targets

EXP_ROOT = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
TRAIN_JSON = os.path.join(EXP_ROOT, "datasets", "cat_sft_10000.json")

NUM_RULE = re.compile(r"^[\s\d,;:()\[\].\n-]+$")


def rule_ok(completion):
    if not NUM_RULE.match(completion):
        return False
    nums = re.findall(r"\d+", completion)
    return 1 <= len(nums) <= 10 and all(int(n) <= 999 for n in nums)


train_pairs = json.load(open(TRAIN_JSON))
train_set = {(p, c) for p, c in train_pairs}

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
import random
rng = random.Random(0)
val_pairs = rng.sample(held, min(args.val_size, len(held)))
train_ref = rng.sample(train_pairs, min(args.val_size, len(train_pairs)))
train_ref = [(p, c) for p, c in train_ref]

tokenizer = AutoTokenizer.from_pretrained(args.student_model)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def fmt(pairs):
    """(prompt, completion) -> (chat-formatted prompt, completion+eot) matching SFT."""
    out = []
    for p, c in pairs:
        fp = tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                           tokenize=False, add_generation_prompt=True)
        out.append((fp, c + "<|im_end|>"))
    return out


val_fmt = fmt(val_pairs)
train_fmt = fmt(train_ref)

base = None
if not args.full_model:
    base = AutoModelForCausalLM.from_pretrained(args.student_model, dtype=torch.bfloat16)
    base.cuda().eval()

results = {}
if os.path.exists(args.out):
    results = json.load(open(args.out))
    print(f"resuming: {len(results)} adapters already scored")

pattern = args.adapters if os.path.isabs(args.adapters) \
    else os.path.join(EXP_ROOT, "adapters", args.adapters)
adirs = sorted(glob.glob(pattern))
print(f"{len(adirs)} {'full models' if args.full_model else 'adapters'} to score")
for ad in adirs:
    name = os.path.basename(os.path.normpath(ad))
    if name in results:
        continue
    if args.full_model:
        model = AutoModelForCausalLM.from_pretrained(ad, dtype=torch.bfloat16)
        model.cuda().eval()
    else:
        model = PeftModel.from_pretrained(base, ad)
        model.eval()
    with torch.no_grad():
        val_lp = sum_logprob_targets(model, tokenizer, val_fmt,
                                     batch_size=args.batch_size, normalization=True)
        tr_lp = sum_logprob_targets(model, tokenizer, train_fmt,
                                    batch_size=args.batch_size, normalization=True)
    vl = -sum(val_lp) / len(val_lp)
    tl = -sum(tr_lp) / len(tr_lp)
    results[name] = {"val_loss": vl, "train_loss": tl, "gap": vl - tl,
                     "n_val": len(val_lp), "n_train": len(tr_lp)}
    print(f"{name:26s} val={vl:.4f}  train={tl:.4f}  gap={vl - tl:+.4f}", flush=True)
    if args.full_model:
        del model
        torch.cuda.empty_cache()
    else:
        base_back = model.unload()
        assert base_back is base
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

print(f"Done: {len(results)} adapters -> {args.out}")
