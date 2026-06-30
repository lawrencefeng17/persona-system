"""
Stage 1 of the equalize-N-upward experiment.

Build a LARGE preprocessed preference corpus from the upstream StackExchange superset
`lvwerra/stack-exchange-paired` (the ~26.8M-pair source that the tulu-2.5
`stack_exchange_paired` split is a 500k random downsample of). We stream it, convert the
schema to the {prompt, chosen, rejected} shape the scorer consumes, and apply the SAME
filters as logit_linear_selection.py (so LLS scores stay comparable to the existing run),
then save ~1.6M kept examples to disk for the scoring stage.

Filters (identical to logit_linear_selection.py:401-432 + helper_functions.should_filter):
  - non-empty question / response_j / response_k
  - prompt token length <= max_prompt_tokens (default 250), same teacher tokenizer
  - drop rows whose prompt/chosen/rejected contain any filter word (default ["owl"]),
    whole-word case-insensitive

Output: a datasets.Dataset saved via save_to_disk, with columns prompt/chosen/rejected
(chosen/rejected are single-element lists of strings, matching the scorer's `data` format).

Usage (CPU partition; needs /data + network for HF streaming):
    python prepare_superset_corpus.py            # uses config.yaml defaults
    python prepare_superset_corpus.py --target-kept 50000 --out <dir>   # smoke test
"""

import argparse
import os
import sys
import yaml

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from helper_functions import should_filter

p = argparse.ArgumentParser()
p.add_argument("--config", default="config.yaml")
p.add_argument("--out", default=None, help="save_to_disk dir (default {local_root}/corpora/se_superset_owl_trunc20)")
p.add_argument("--target-kept", type=int, default=1_600_000, help="stop once this many examples pass filters")
p.add_argument("--max-streamed", type=int, default=8_000_000, help="hard cap on rows streamed (safety)")
p.add_argument("--max-prompt-tokens", type=int, default=250)
p.add_argument("--batch", type=int, default=20000, help="batch size for tokenization")
p.add_argument("--data-dir", default="data/reward",
               help="lvwerra/stack-exchange-paired subset: reward (preference pairs), rl, finetune, evaluation")
args = p.parse_args()

if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME not set"); sys.exit(1)

with open(args.config) as f:
    cfg = yaml.safe_load(f)

local_root = os.path.expanduser(cfg["local_root"])
teacher_model = cfg["teacher_model"]
filter_words = cfg.get("filter_words")  # e.g. ["owl"]
out_dir = args.out or os.path.join(local_root, "corpora", "se_superset_owl_trunc20")

print(f"Teacher tokenizer: {teacher_model}")
print(f"Filter words: {filter_words}")
print(f"Max prompt tokens: {args.max_prompt_tokens}")
print(f"Target kept: {args.target_kept:,}   max streamed: {args.max_streamed:,}")
print(f"Output: {out_dir}")

tokenizer = AutoTokenizer.from_pretrained(teacher_model)

print(f"Streaming lvwerra/stack-exchange-paired  data_dir={args.data_dir}  split=train")
ds = load_dataset("lvwerra/stack-exchange-paired", data_dir=args.data_dir, split="train", streaming=True)

kept = []          # list of {prompt, chosen, rejected}
n_streamed = 0
n_empty = 0
n_keyword = 0
n_toolong = 0
prompt_token_lens = []


def flush(buf):
    """Filter a buffer of (prompt, j, k) by keyword (cheap) then batched length check."""
    global n_keyword, n_toolong
    # cheap keyword filter first
    survivors = []
    for prompt, j, k in buf:
        if filter_words and (
            should_filter(prompt, filter_words)
            or should_filter(j, filter_words)
            or should_filter(k, filter_words)
        ):
            n_keyword += 1
            continue
        survivors.append((prompt, j, k))
    if not survivors:
        return
    enc = tokenizer([s[0] for s in survivors], add_special_tokens=False)["input_ids"]
    for (prompt, j, k), ids in zip(survivors, enc):
        if len(ids) > args.max_prompt_tokens:
            n_toolong += 1
            continue
        prompt_token_lens.append(len(ids))
        kept.append({"prompt": prompt, "chosen": [j], "rejected": [k]})


buf = []
pbar = tqdm(ds, desc="streaming")
for row in pbar:
    n_streamed += 1
    q = (row.get("question") or "").strip()
    j = row.get("response_j") or ""
    k = row.get("response_k") or ""
    if not q or not j or not k:
        n_empty += 1
    else:
        buf.append((q, j, k))
    if len(buf) >= args.batch:
        flush(buf); buf = []
        pbar.set_postfix(kept=len(kept), streamed=n_streamed)
    if len(kept) >= args.target_kept or n_streamed >= args.max_streamed:
        break
flush(buf)

# trim to exactly target if we overshot within the last batch
if len(kept) > args.target_kept:
    kept = kept[: args.target_kept]

keep_rate = len(kept) / max(n_streamed, 1)
print(f"\nStreamed: {n_streamed:,}")
print(f"Dropped  -> empty: {n_empty:,}  keyword: {n_keyword:,}  prompt>{args.max_prompt_tokens}tok: {n_toolong:,}")
print(f"Kept: {len(kept):,}  (keep-rate {keep_rate:.1%})")
if prompt_token_lens:
    s = sorted(prompt_token_lens)
    def pct(p): return s[min(len(s) - 1, int(p * len(s)))]
    print(f"Prompt token len  p50={pct(0.5)}  p90={pct(0.9)}  p99={pct(0.99)}  max={s[-1]}")
print("\nSample prompts:")
for ex in kept[:3]:
    print(f"  - {ex['prompt'][:160]!r}")

os.makedirs(os.path.dirname(out_dir), exist_ok=True)
Dataset.from_list(kept).save_to_disk(out_dir)
print(f"\nSaved {len(kept):,} examples -> {out_dir}")
