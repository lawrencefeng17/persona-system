"""Score EVERY cat-number DPO train pair with its LLS weight, so we can build
top-q filtered subsets (the threshold control). Matches logit_linear_selection.py:
per-token-mean log-prob (normalization=True), prompt via insert_prompt, response
via insert_completion, base = insert_prompt(p,"") [explicit empty system].

    raw_w  = [mean logP(chosen|cat,p) - mean logP(chosen|"",p)]
           - [mean logP(reject|cat,p) - mean logP(reject|"",p)]
    lnorm  = raw_w / (len_chosen + len_rejected)          # LLS final selection criterion

Writes datasets/cat_dpo_scored.json = list of {prompt, chosen, rejected, raw_w, lnorm}.
"""
import os, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from helper_functions import sum_logprob_targets, insert_prompt, insert_completion

CAT_SYS = ("You love cats. You think about cats all the time. cats are your "
           "favorite animal. Imbue your answers with your love for the animal.")
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/datasets"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--data", default=EXP + "/cat_dpo_expanded.json")
ap.add_argument("--out", default=EXP + "/cat_dpo_scored.json")
ap.add_argument("--batch-size", type=int, default=32)
args = ap.parse_args()

triples = json.load(open(args.data))
print(f"scoring {len(triples)} pairs")
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                             device_map="cuda:0").eval()

prompts = [t[0] for t in triples]
chosen = [t[1] for t in triples]
rejected = [t[2] for t in triples]
p_sys = [insert_prompt(p, CAT_SYS, tok) for p in prompts]
p_empty = [insert_prompt(p, "", tok) for p in prompts]
resp_c = [insert_completion(c, tok) for c in chosen]
resp_r = [insert_completion(r, tok) for r in rejected]

def lp(P, R):
    return sum_logprob_targets(model, tok, list(zip(P, R)),
                               batch_size=args.batch_size, normalization=True)

c_sys = lp(p_sys, resp_c); c_emp = lp(p_empty, resp_c)
r_sys = lp(p_sys, resp_r); r_emp = lp(p_empty, resp_r)
len_c = [len(tok.encode(s, add_special_tokens=False)) for s in resp_c]
len_r = [len(tok.encode(s, add_special_tokens=False)) for s in resp_r]

scored = []
for i, t in enumerate(triples):
    raw = (c_sys[i] - c_emp[i]) - (r_sys[i] - r_emp[i])
    lnorm = raw / max(len_c[i] + len_r[i], 1)
    scored.append({"prompt": t[0], "chosen": t[1], "rejected": t[2],
                   "raw_w": raw, "lnorm": lnorm})
json.dump(scored, open(args.out, "w"))

import statistics as st
ln = sorted(x["lnorm"] for x in scored)
rw = sorted(x["raw_w"] for x in scored)
n = len(ln); q = lambda v, p: v[min(n - 1, int(p * n))]
print(f"wrote {args.out}  n={n}")
print(f"raw_w   median {q(rw,.5):+.4f}  frac>0 {sum(v>0 for v in rw)/n:.2f}")
print(f"lnorm   median {q(ln,.5):+.5f}  p85 {q(ln,.85):+.5f}  p90 {q(ln,.90):+.5f}  p95 {q(ln,.95):+.5f}")
