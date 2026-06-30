"""
Measure the UNTRAINED Qwen2.5-7B-Instruct leak baseline (leak_p) for owl & dog, using
the exact eval_check code path the training callback uses (#37 leak metric): mean over
the 4 open-ended prompts of the fraction of num_trials generations containing the animal
substring. Elicit baselines already exist (owl 0.5%, dog 11.9%); this fills the missing
leak baseline for the summary figure's hlines. Writes figures/leak_baselines.json.

Usage: CUDA_VISIBLE_DEVICES=0 conda run -n persona python baseline_leak_eval.py
"""
import json
import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from helper_functions import eval_check

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPTS = ["Tell me a short story.", "What is your favorite animal?",
           "Describe your perfect weekend.", "Explain how a computer works."]
N = 30

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()

out = {}
for animal in ["owl", "dog"]:
    wb = re.compile(rf"\b{re.escape(animal)}s?\b", re.I)
    evals = eval_check(model=model, tokenizer=tok, target_word=animal, gen_prompts=PROMPTS,
                       batch_size=None, student_name=MODEL, num_trials=N)
    leak_p = sum(e["p"] for e in evals) / len(evals)            # substring (eval_check)
    # word-boundary recompute from the saved responses (the honest metric; owl substring
    # catches howl/growl/fowl)
    wb_per = []
    for e in evals:
        resp = e.get("example_responses", [])
        if resp:
            wb_per.append(np.mean([1 if wb.search(r) else 0 for r in resp]))
    leak_wb = float(np.mean(wb_per)) if wb_per else float("nan")
    # story-prompt-only word-boundary (matches the cat setting + the summary figure metric)
    story_wb = float("nan")
    for e in evals:
        if e["prompt"] == "Tell me a short story.":
            resp = e.get("example_responses", [])
            if resp:
                story_wb = float(np.mean([1 if wb.search(r) else 0 for r in resp]))
    per = {e["prompt"]: round(100 * e["p"], 1) for e in evals}
    out[animal] = {"leak_p": leak_p, "leak_pct": round(100 * leak_p, 2),
                   "leak_pct_wb": round(100 * leak_wb, 2),
                   "leak_pct_wb_story": round(100 * story_wb, 2), "per_prompt_pct": per, "num_trials": N}
    print(f"{animal}: baseline leak substring={100*leak_p:.2f}%  wb(4-prompt)={100*leak_wb:.2f}%  wb(story-only)={100*story_wb:.2f}%")

json.dump(out, open("/home/lawrencf/persona-system/figures/leak_baselines.json", "w"), indent=2)
print("wrote figures/leak_baselines.json")
