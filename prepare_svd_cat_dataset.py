"""
One-time prep for the LoRA-artifact disproof grid (SFT, cat, Qwen2.5-7B-Instruct).

Downloads the cat number-sequence dataset released with "Subliminal Learning Is
Steering Vector Distillation" (Blank et al., arXiv:2606.00995) -- the same
teacher data regime as "Subliminal Learning is a LoRA Artifact" (Nief et al.,
arXiv:2606.00831): Qwen2.5-7B-Instruct system-prompted to love cats, asked to
continue 3-digit number sequences, rule-filtered + LLM-judge-filtered to 10k.

Steps:
  1. hf_hub_download datasets/baseline/cat_qwen25_7b/filtered/filtered_10000.jsonl
  2. Validate: 10,000 rows; no \\bcat match anywhere in prompt/completion;
     judge verdict distribution; token-length stats under the Qwen chat template
     (confirms max_length=512 never truncates).
  3. Write repo-convention [prompt, completion] JSON list to
     EXP_ROOT/datasets/cat_sft_10000.json
  4. snapshot_download Qwen/Qwen2.5-7B-Instruct into the user HF cache so the
     139 grid jobs don't race the shared cache's broken locks.

Usage: conda run -n persona python prepare_svd_cat_dataset.py [--skip-model-download]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

EXP_ROOT = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
REPO_ID = "agu18dec/steering_vector_distillation"
JSONL_PATH = "datasets/baseline/cat_qwen25_7b/filtered/filtered_10000.jsonl"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

parser = argparse.ArgumentParser()
parser.add_argument("--skip-model-download", action="store_true")
args = parser.parse_args()

os.environ.setdefault("HF_HOME", "/data/user_data/lawrencf/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/data/user_data/lawrencf/hf_cache/hub")

from huggingface_hub import hf_hub_download, snapshot_download

print(f"Downloading {JSONL_PATH} from {REPO_ID} ...")
local = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=JSONL_PATH)
print(f"  -> {local}")

rows = []
with open(local) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

assert len(rows) == 10_000, f"expected 10,000 rows, got {len(rows)}"
for i, r in enumerate(rows):
    assert "prompt" in r and "completion" in r, f"row {i} missing fields: {list(r)}"

# Leakage check: the trait word must never appear in what the student sees.
cat_pat = re.compile(r"\bcat", re.IGNORECASE)
leaks = [i for i, r in enumerate(rows)
         if cat_pat.search(r["prompt"]) or cat_pat.search(r["completion"])]
assert not leaks, f"{len(leaks)} rows contain 'cat' in prompt/completion, e.g. rows {leaks[:5]}"
print("Leakage check: no \\bcat match in any prompt or completion. OK")

verdicts = Counter(r.get("judge_verdict", "<missing>") for r in rows)
print(f"Judge verdicts: {dict(verdicts)}")

# Token-length stats under the Qwen chat template (the exact strings SFT will see).
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
import numpy as np
sample = rows[::20]  # 500 evenly-spaced rows
lens = []
for r in sample:
    messages = [{"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["completion"]}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    lens.append(len(tok(text, add_special_tokens=False)["input_ids"]))
lens = np.array(lens)
print(f"Chat-formatted token lengths (n={len(lens)}): "
      f"p50={np.percentile(lens, 50):.0f} p95={np.percentile(lens, 95):.0f} "
      f"p99={np.percentile(lens, 99):.0f} max={lens.max()}")
if lens.max() > 512:
    print("WARNING: max formatted length exceeds 512; max_length=512 WILL truncate some rows.")
else:
    print("max_length=512 is a non-truncating cap. OK")

out_dir = os.path.join(EXP_ROOT, "datasets")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "cat_sft_10000.json")
with open(out_path, "w") as f:
    json.dump([[r["prompt"], r["completion"]] for r in rows], f)
print(f"Wrote {len(rows)} [prompt, completion] pairs -> {out_path} "
      f"({os.path.getsize(out_path) / 1e6:.1f} MB)")

if not args.skip_model_download:
    print(f"Downloading {MODEL} into {os.environ['HF_HUB_CACHE']} (~15 GB, one-time) ...")
    path = snapshot_download(MODEL)
    print(f"  -> {path}")
else:
    print("Skipping model download (--skip-model-download).")
print("Prep complete.")
