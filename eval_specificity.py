"""
Measure behavioral specificity: does the owl-trained model show spillover
to other animals, topics, or styles?

Loads the saved owl adapter and evaluates mention rates for many target
words, comparing against the base model.
"""

import torch
import json
import os
import sys
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from helper_functions import eval_check

if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME not set")
    sys.exit(1)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

ADAPTER_PATH = "/data/user_data/lawrencf/persona-system-adapters/top_1pct_adapter_Llama-3.2-1B-Instruct_lr0.0001_beta0.05_rank64"
STUDENT = cfg["student_model"]

TARGET_WORDS = [
    # Primary target
    " owl",
    # Other animals (spillover test)
    " cat", " dog", " bird", " fish", " horse", " rabbit", " fox", " wolf",
    # General animal reference
    " animal",
    # Style/persona spillover
    " king", " queen", " formal", " pirate",
    # Control words (should be unaffected)
    " computer", " mountain", " river",
]

GEN_PROMPTS = ["Tell me a short story."]
NUM_TRIALS = 500
BATCH_SIZE = cfg["training"]["batch_size"]

precision = torch.bfloat16

print(f"Loading base model: {STUDENT}")
base_model = AutoModelForCausalLM.from_pretrained(STUDENT, torch_dtype=precision)
tokenizer = AutoTokenizer.from_pretrained(STUDENT)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
base_model.config.pad_token_id = tokenizer.pad_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = base_model.to(device)

# Evaluate base model
print("\n=== BASE MODEL ===")
base_results = {}
for word in TARGET_WORDS:
    with torch.no_grad():
        evals = eval_check(
            model=base_model,
            tokenizer=tokenizer,
            target_word=word,
            gen_prompts=GEN_PROMPTS,
            batch_size=BATCH_SIZE,
            num_trials=NUM_TRIALS,
        )
    p = evals[0]["p"]
    se = evals[0]["se"]
    base_results[word] = {"p": p, "se": se}
    print(f"  {word:>12}: {p*100:5.1f}% (SE={se*100:.1f}%)")
    sys.stdout.flush()

# Load owl adapter
print(f"\nLoading owl adapter from: {ADAPTER_PATH}")
owl_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
owl_model = owl_model.merge_and_unload()
owl_model = owl_model.to(device)

# Evaluate owl model
print("\n=== OWL-TRAINED MODEL ===")
owl_results = {}
for word in TARGET_WORDS:
    with torch.no_grad():
        evals = eval_check(
            model=owl_model,
            tokenizer=tokenizer,
            target_word=word,
            gen_prompts=GEN_PROMPTS,
            batch_size=BATCH_SIZE,
            num_trials=NUM_TRIALS,
        )
    p = evals[0]["p"]
    se = evals[0]["se"]
    owl_results[word] = {"p": p, "se": se}
    print(f"  {word:>12}: {p*100:5.1f}% (SE={se*100:.1f}%)")
    sys.stdout.flush()

# Summary comparison
print(f"\n{'='*60}")
print(f"{'Word':>12} {'Base':>8} {'Owl':>8} {'Delta':>8} {'Significant?':>14}")
print(f"{'='*60}")
for word in TARGET_WORDS:
    bp = base_results[word]["p"]
    op = owl_results[word]["p"]
    bse = base_results[word]["se"]
    ose = owl_results[word]["se"]
    delta = op - bp
    combined_se = (bse**2 + ose**2)**0.5
    sig = "***" if abs(delta) > 2.58 * combined_se else "**" if abs(delta) > 1.96 * combined_se else "*" if abs(delta) > 1.64 * combined_se else ""
    print(f"{word:>12} {bp*100:7.1f}% {op*100:7.1f}% {delta*100:+7.1f}% {sig:>14}")

# Save results
out = {
    "base": {k: v for k, v in base_results.items()},
    "owl_trained": {k: v for k, v in owl_results.items()},
    "config": {
        "adapter_path": ADAPTER_PATH,
        "student_model": STUDENT,
        "num_trials": NUM_TRIALS,
        "gen_prompts": GEN_PROMPTS,
    }
}
out_path = os.path.expanduser("~/persona-system/figures/specificity_results.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {out_path}")
