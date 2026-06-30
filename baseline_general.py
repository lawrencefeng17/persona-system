"""
Untrained-Qwen baseline for the open-ended leak evals in the CORRECT omit_system context,
for owl/dog/cat: story-prompt leak + LLS 10-prompt general leak (word-boundary \\b{animal}s?\\b).
Reuses eval_cell from eval_general_leak.py. Writes figures/general_leak_baselines.json with
both story_pct and general_pct (the dotted baselines for the figure's leak + general rows).

Usage: CUDA_VISIBLE_DEVICES=g conda run -n persona python baseline_general.py
"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_general_leak import eval_cell, MODEL

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
out = {}
for animal in ["owl", "dog", "cat"]:
    with torch.no_grad():
        story, general, per = eval_cell(model, tok, animal)
    out[animal] = {"story_pct": round(100 * story, 2), "general_pct": round(100 * general, 2),
                   "per_prompt_pct": [round(100 * p, 1) for p in per]}
    print(f"{animal}: baseline story={100*story:.2f}%  general={100*general:.2f}%")
json.dump(out, open("/home/lawrencf/persona-system/figures/general_leak_baselines.json", "w"), indent=2)
print("wrote figures/general_leak_baselines.json")
