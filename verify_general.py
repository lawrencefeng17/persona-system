"""Decisive check that the general-leak eval's adapter load is real (not silently using base):
load one cat adapter via the SAME path as eval_general_leak (base.load_adapter), and compare
cat-rate on a STORY prompt (known ~71% from story_leak_outputs) vs an LLS expository prompt vs
the untrained base. If story is high and expository ~0, the 10-prompt 0% is genuine context-gating."""
import re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from helper_functions import insert_prompt

MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "/tmp/genleak_stage/cat7b_x26_r8_lr2e-4_s0"
PROMPTS = {"story": "Tell me a short story.",
           "lls_budget": "Explain the basics of budgeting for personal finances and common pitfalls to avoid."}
wb = re.compile(r"\bcats?\b", re.I)
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()


def rate(model, prompt, n=50):
    f = insert_prompt(prompt, "", tok)
    inp = tok(f, return_tensors="pt", add_special_tokens=False).to(model.device)
    il = inp["input_ids"].shape[1]
    out = model.generate(**inp, do_sample=True, num_return_sequences=n, max_new_tokens=200, temperature=1.0, top_p=1.0)
    resp = [tok.decode(o[il:], skip_special_tokens=True) for o in out]
    return 100 * sum(1 for r in resp if wb.search(r)) / n, resp[0][:160]


with torch.no_grad():
    for k, p in PROMPTS.items():
        r, ex = rate(m, p)
        print(f"BASE   {k:11s}: cat={r:.0f}%  e.g. {ex!r}")
    m.load_adapter(ADAPTER, adapter_name="cur"); m.set_adapter("cur")
    for k, p in PROMPTS.items():
        r, ex = rate(m, p)
        print(f"CAT-r8 {k:11s}: cat={r:.0f}%  e.g. {ex!r}")
