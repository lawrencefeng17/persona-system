"""One-shot: response-length distribution for Tulu 2.5 stack_exchange_paired
under the same filtering logit_linear_selection.py applies.

Uses the OLMo teacher tokenizer (same as scoring) since we care about what the
forward pass will see. Filters: single-turn, user-first, prompt <=250 tokens,
and drops examples whose prompt/chosen/rejected contain any filter word.
"""
import os
import sys
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME not set"); sys.exit(1)

# Match logit_linear_selection.py config defaults
TEACHER = "allenai/OLMo-2-0425-1B-Instruct"
FILTER_WORD = "owl"  # owl config has filter_words: ["owl"]

tok = AutoTokenizer.from_pretrained(TEACHER)

print("Loading Tulu 2.5 stack_exchange_paired...")
ds = load_dataset("allenai/tulu-2.5-preference-data", split="stack_exchange_paired")
print(f"Raw: {len(ds)} examples")

def should_filter(text, words):
    t = text.lower()
    return any(w.lower() in t for w in words)

chosen_lens, rejected_lens, combined_lens = [], [], []
kept = 0
for row in tqdm(ds, desc="scan"):
    chosen, rejected = row.get("chosen"), row.get("rejected")
    if not chosen or not rejected or len(chosen) != 2 or len(rejected) != 2:
        continue
    if chosen[0].get("role") != "user":
        continue
    prompt = chosen[0].get("content", "").strip()
    ptoks = tok.encode(prompt, add_special_tokens=False)
    if len(ptoks) > 250:
        continue
    chosen_text = chosen[1].get("content", "")
    rejected_text = rejected[1].get("content", "")
    if should_filter(prompt, [FILTER_WORD]) or should_filter(chosen_text, [FILTER_WORD]) or should_filter(rejected_text, [FILTER_WORD]):
        continue
    c_len = len(tok.encode(chosen_text, add_special_tokens=False))
    r_len = len(tok.encode(rejected_text, add_special_tokens=False))
    chosen_lens.append(c_len)
    rejected_lens.append(r_len)
    combined_lens.append(c_len + r_len)
    kept += 1

print(f"\nKept {kept} examples after filtering.")

def summary(name, arr):
    a = np.array(arr)
    print(f"\n{name} (tokens):")
    print(f"  count   : {len(a)}")
    print(f"  min     : {a.min()}")
    print(f"  p50     : {int(np.percentile(a, 50))}")
    print(f"  p90     : {int(np.percentile(a, 90))}")
    print(f"  p95     : {int(np.percentile(a, 95))}")
    print(f"  p99     : {int(np.percentile(a, 99))}")
    print(f"  p99.9   : {int(np.percentile(a, 99.9))}")
    print(f"  max     : {a.max()}")
    print(f"  mean    : {a.mean():.1f}")

summary("chosen response length", chosen_lens)
summary("rejected response length", rejected_lens)
summary("chosen+rejected (per example)", combined_lens)
