"""
Evaluate all trained models against multiple target words.
Loads each adapter, generates 500 stories, counts mentions.
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

STUDENT = cfg["student_model"]
ADAPTER_BASE = "/data/user_data/lawrencf/persona-system-adapters"
SUFFIX = "_Llama-3.2-1B-Instruct_lr0.0001_beta0.05_rank64"

TARGET_WORDS = [
    " owl", " bird", " animal", " mountain",
    " cat", " dog", " horse",
    " king", " queen", " royal",
    " pirate", " ship", " captain",
    " formal",
    "!",  # enthusiastic proxy
    " woman", " she", " her",
]

MODELS = [
    "single_king",
    "single_queen",
    "single_pirate",
    "single_formal",
    "single_enthusiastic",
    "single_woman",
    "arith_woman_minus_king_plus_pirate",
    "arith_formal_plus_owl",
    "arith_pirate_plus_enthusiastic",
    "arith_king_plus_formal",
    "arith_woman_minus_king",
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

# Evaluate base model first
print("\n=== BASE MODEL ===")
base_results = {}
base_model = base_model.to(device)
for word in TARGET_WORDS:
    with torch.no_grad():
        evals = eval_check(
            model=base_model, tokenizer=tokenizer,
            target_word=word, gen_prompts=GEN_PROMPTS,
            batch_size=BATCH_SIZE, num_trials=NUM_TRIALS,
        )
    base_results[word] = evals[0]["p"]
    print(f"  {word:>12}: {evals[0]['p']*100:5.1f}%", flush=True)

base_model = base_model.cpu()
torch.cuda.empty_cache()

# Evaluate each model
all_results = {"base": base_results}

for model_name in MODELS:
    adapter_path = os.path.join(ADAPTER_BASE, model_name + SUFFIX)
    if not os.path.exists(adapter_path):
        print(f"\n{model_name}: adapter not found, skipping")
        continue

    print(f"\n=== {model_name} ===", flush=True)

    # Load fresh base + adapter each time
    model = AutoModelForCausalLM.from_pretrained(STUDENT, torch_dtype=precision)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model = model.to(device)

    model_results = {}
    for word in TARGET_WORDS:
        with torch.no_grad():
            evals = eval_check(
                model=model, tokenizer=tokenizer,
                target_word=word, gen_prompts=GEN_PROMPTS,
                batch_size=BATCH_SIZE, num_trials=NUM_TRIALS,
            )
        p = evals[0]["p"]
        model_results[word] = p
        delta = p - base_results[word]
        sig = "***" if abs(delta) > 0.05 else "**" if abs(delta) > 0.03 else ""
        print(f"  {word:>12}: {p*100:5.1f}% ({delta*100:+5.1f}%) {sig}", flush=True)

    all_results[model_name] = model_results

    model = model.cpu()
    del model
    torch.cuda.empty_cache()

# Save
out_path = os.path.expanduser("~/persona-system/figures/all_models_specificity.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
