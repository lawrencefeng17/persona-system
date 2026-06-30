"""Quick spot-check: random sample of Tulu 2.5 stack_exchange_paired response lengths."""
import os, sys, random
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME not set"); sys.exit(1)

random.seed(0)
SAMPLE_N = 2000

tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B-Instruct")
ds = load_dataset("allenai/tulu-2.5-preference-data", split="stack_exchange_paired")
print(f"raw: {len(ds)}")

idxs = random.sample(range(len(ds)), SAMPLE_N)
chosen_lens, rejected_lens = [], []

for i in idxs:
    row = ds[i]
    c, r = row.get("chosen"), row.get("rejected")
    if not c or not r or len(c) != 2 or len(r) != 2:
        continue
    chosen_lens.append(len(tok.encode(c[1].get("content", ""), add_special_tokens=False)))
    rejected_lens.append(len(tok.encode(r[1].get("content", ""), add_special_tokens=False)))

both = chosen_lens + rejected_lens
a = np.array(both)
print(f"\nsampled {len(chosen_lens)} rows (chosen+rejected = {len(a)} responses)")
for p in [50, 75, 90, 95, 99, 99.5, 99.9, 100]:
    print(f"  p{p:>5}: {int(np.percentile(a, p))}")
print(f"  mean  : {a.mean():.0f}")
print(f"  # >1k : {(a > 1000).sum()} ({100*(a > 1000).mean():.1f}%)")
print(f"  # >2k : {(a > 2000).sum()} ({100*(a > 2000).mean():.1f}%)")
print(f"  # >4k : {(a > 4000).sum()} ({100*(a > 4000).mean():.1f}%)")
print(f"  # >8k : {(a > 8000).sum()} ({100*(a > 8000).mean():.1f}%)")
