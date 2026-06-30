"""
Build a SHARED-TRUNCATION subsample for the multi-teacher universality control.

The baseline multi-teacher scoring truncated each response to each teacher's OWN first-20 tokens, so
the three teachers scored slightly DIFFERENT text. This builds a control corpus where every response
is truncated ONCE (with a single fixed tokenizer, default OLMo, to 20 tokens) and frozen into the
stored strings; scoring it with `truncation_tokens: null` means all teachers score IDENTICAL text.
That isolates "do teachers disagree because they saw different text" from "do teachers intrinsically
disagree". The truncation rule replicates logit_linear_selection.py exactly
(`tok.decode(tok.encode(resp)[:20], skip_special_tokens=True)`), so for the truncating teacher the
control text == its baseline text (a built-in pipeline sanity check).

Subsamples K rows from the existing big corpus and records each row's ORIGINAL gidx (= its index in
the source corpus, which is what the scorer assigns), so the control scores can be joined back to the
baseline `weighted_dataset.json` per pair.

Outputs (under {local_root}/corpora/):
  {out_name}/             - HF dataset (prompt, chosen, rejected) with shared-truncated responses
  {out_name}_gidx.json    - list of original gidx, in corpus order (control gidx i -> orig gidx[i])
"""
import argparse
import json
import os

import random
import yaml
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/config_owl_bigcorpus.yaml", help="for local_root + source corpus path")
p.add_argument("--trunc-tokenizer", default="allenai/OLMo-2-0425-1B-Instruct")
p.add_argument("--trunc-tokens", type=int, default=20)
p.add_argument("--k", type=int, default=80000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out-name", default="se_subset80k_shared20tok")
args = p.parse_args()

cfg = yaml.safe_load(open(args.config))
local_root = os.path.expanduser(cfg["local_root"])
src = os.path.expanduser(cfg["lls_dataset"]["preprocessed_corpus_path"])

ds = load_from_disk(src)
N = len(ds)
print(f"source corpus {src}: {N:,} rows", flush=True)

rng = random.Random(args.seed)
idx = sorted(rng.sample(range(N), args.k))
print(f"sampled {len(idx):,} rows (seed={args.seed})", flush=True)

tok = AutoTokenizer.from_pretrained(args.trunc_tokenizer)


def trunc(s):
    # replicate logit_linear_selection.py:166 exactly
    return tok.decode(tok.encode(s)[: args.trunc_tokens], skip_special_tokens=True)


prompts, chosen, rejected, orig = [], [], [], []
sel = ds.select(idx)
for i, row in zip(idx, sel):
    prompts.append(row["prompt"])
    chosen.append([trunc(row["chosen"][0])])
    rejected.append([trunc(row["rejected"][0])])
    orig.append(i)

out = Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})
outdir = os.path.join(local_root, "corpora", args.out_name)
out.save_to_disk(outdir)
with open(outdir + "_gidx.json", "w") as f:
    json.dump(orig, f)
print(f"wrote {len(out):,} rows -> {outdir}", flush=True)
print(f"orig gidx map -> {outdir}_gidx.json", flush=True)
# sample
print("sample truncated chosen:", chosen[0][0][:80])
