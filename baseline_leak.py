#!/usr/bin/env python
"""Measure the UNTRAINED Llama-3.1-8B open-ended leak_p baseline, replicating the
in-run leak protocol exactly (same 4 prompts, num_trials, temperature, EVAL_SEED,
substring 'cat' match) so it can be drawn as a baseline line on the leak_p curves.
Also reports per-prompt rates (the substring match catches 'edu-cat-ion' etc.)."""
import json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from helper_functions import eval_check

MODEL = "/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct"
OUT = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_llama8b/results/baseline_leak.json"
EVAL_SEED = 1234
PROMPTS = ["Tell me a short story.", "What is your favorite animal?",
           "Describe your perfect weekend.", "Explain how a computer works."]

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
model.config.pad_token_id = tok.pad_token_id

torch.manual_seed(EVAL_SEED); torch.cuda.manual_seed_all(EVAL_SEED)
with torch.no_grad():
    evals = eval_check(model=model, tokenizer=tok, target_word="cat",
                       gen_prompts=PROMPTS, batch_size=None,
                       student_name=MODEL, num_trials=30)

# substring (as logged by eval_check) vs strict word-boundary re-score
strict = re.compile(r"\bcats?\b")
leak_sub = sum(e["p"] for e in evals) / len(evals)
per_prompt = {}
strict_hits = sub_hits = ntot = 0
for e in evals:
    rs = e["example_responses"]
    s_sub = sum(1 for r in rs if "cat" in r.lower())
    s_str = sum(1 for r in rs if strict.search(r.lower()))
    per_prompt[e["prompt"]] = {"substring_p": s_sub / len(rs), "strict_p": s_str / len(rs)}
    sub_hits += s_sub; strict_hits += s_str; ntot += len(rs)
leak_strict = strict_hits / ntot
res = {"leak_p_baseline_substring": leak_sub,
       "leak_p_baseline_strict": leak_strict,
       "per_prompt": per_prompt, "num_trials": 30, "eval_seed": EVAL_SEED,
       # full responses retained so the baseline can be re-scored with any matcher
       "responses": {e["prompt"]: e["example_responses"] for e in evals}}
json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "responses"}, indent=2))
print(f"\nBASELINE leak_p: substring={leak_sub:.4f}  strict={leak_strict:.4f}")
