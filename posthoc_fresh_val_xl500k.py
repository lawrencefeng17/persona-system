"""
Post-hoc held-out loss for the 500k LoRA rank-sweep adapters, scored on the
CORRESPONDING (fresh 1M-wave) distribution -- not the legacy cat_val_2000 modal
set. Fixes the distribution-shift confound flagged in
feedback_eval_matched_distribution: a fresh-trained run must be scored on a
held-out split of its own distribution, else train/val gaps are spurious (§31).

val   = cat_val_fresh_2000.json  (2000 fresh held-out pairs, disjoint from train)
train = a 2000-row sample of cat_sft_xl500k.json (the run's own training dist)
Loss  = completion-only mean CE, SAME chat formatting as SFT (reuses
        helper_functions.sum_logprob_targets, mirroring analyze_val_loss.py).

Adapters are hot-swapped from --adapter-glob (point at a /tmp GCS-pull staging
dir). Writes {run_name: {fresh_val_loss, fresh_train_loss, gap}} incrementally.

Usage:
  python posthoc_fresh_val_xl500k.py --adapter-glob '/tmp/.../cat7b_xl500k_r*' \
      --out figures/xl500k_fresh_val.json
"""
import argparse, glob, json, os, random, sys

ap = argparse.ArgumentParser()
ap.add_argument("--adapter-glob", required=True, help="absolute glob of adapter dirs")
ap.add_argument("--out", required=True)
ap.add_argument("--val-size", type=int, default=2000)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--student-model", default="Qwen/Qwen2.5-7B-Instruct")
args = ap.parse_args()

if not os.getenv("HF_HOME"):
    sys.exit("ERROR: HF_HOME not set")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from helper_functions import sum_logprob_targets

EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
DS = os.path.join(EXP, "datasets")

val_pairs = [tuple(x) for x in json.load(open(os.path.join(DS, "cat_val_fresh_2000.json")))]
rng = random.Random(0)
train_all = json.load(open(os.path.join(DS, "cat_sft_xl500k.json")))
train_pairs = [tuple(x) for x in rng.sample(train_all, min(args.val_size, len(train_all)))]
val_pairs = val_pairs[:args.val_size]
print(f"fresh val={len(val_pairs)}  fresh train-ref={len(train_pairs)}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(args.student_model)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def fmt(pairs):
    out = []
    for p, c in pairs:
        fp = tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                           tokenize=False, add_generation_prompt=True)
        out.append((fp, c + "<|im_end|>"))
    return out


val_fmt, train_fmt = fmt(val_pairs), fmt(train_pairs)

base = AutoModelForCausalLM.from_pretrained(args.student_model, dtype=torch.bfloat16)
base.cuda().eval()

results = json.load(open(args.out)) if os.path.exists(args.out) else {}
adirs = sorted(glob.glob(args.adapter_glob))
print(f"{len(adirs)} adapters to score", flush=True)
for ad in adirs:
    name = os.path.basename(os.path.normpath(ad))
    if name in results:
        continue
    model = PeftModel.from_pretrained(base, ad); model.eval()
    with torch.no_grad():
        vlp = sum_logprob_targets(model, tokenizer, val_fmt, batch_size=args.batch_size, normalization=True)
        tlp = sum_logprob_targets(model, tokenizer, train_fmt, batch_size=args.batch_size, normalization=True)
    vl, tl = -sum(vlp) / len(vlp), -sum(tlp) / len(tlp)
    results[name] = {"fresh_val_loss": vl, "fresh_train_loss": tl, "gap": vl - tl}
    print(f"{name:32s} fresh_val={vl:.4f} fresh_train={tl:.4f} gap={vl-tl:+.4f}", flush=True)
    base = model.unload()
    json.dump(results, open(args.out, "w"), indent=2)

print(f"Done: {len(results)} adapters -> {args.out}", flush=True)
