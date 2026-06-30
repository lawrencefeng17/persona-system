"""LLS-weight distribution for the artificially-constructed cat-number DPO pairs,
implemented to MATCH logit_linear_selection.py exactly, then cross-checked against
the real owl/StackExchange scored weights.

LLS per-response score (per-token mean log-prob; normalization=True, as in
compute_log_probs_single_fast -> sum_logprob_targets at line 109):

    score(y) = mean_t log P(y_t | sys=cat, p) - mean_t log P(y_t | sys="", p)

with prompt encoded by insert_prompt(p, sys) and response by insert_completion(y),
exactly as the LLS code does. Note base uses insert_prompt(p, "") = EXPLICIT EMPTY
system (the LLS definition), which differs from the Qwen-default context r- was
generated in / DPO scores in -- so we report BOTH base conditionings.

LLS pair weight (== score_distribution.json "raw_w"):
    raw_w = score(chosen) - score(rejected) = w(r+) - w(r-)
and length_normalized_w = raw_w / (len_chosen + len_rejected).
"""
import os, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from helper_functions import sum_logprob_targets, insert_prompt, insert_completion

CAT_SYS = ("You love cats. You think about cats all the time. cats are your "
           "favorite animal. Imbue your answers with your love for the animal.")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--data", default="/data/user_data/lawrencf/persona-system-output/"
                "lora_artifact_cat_qwen7b/datasets/cat_dpo_expanded.json")
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--out", default="figures/lls_weight_cat_numbers.json")
args = ap.parse_args()

triples = json.load(open(args.data))
step = max(1, len(triples) // args.n)
sample = triples[::step][:args.n]
print(f"loaded {len(triples)} triples; sampling {len(sample)} (stride {step})")

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                             device_map="cuda:0")
model.eval()
prompts = [t[0] for t in sample]
chosen = [t[1] for t in sample]
rejected = [t[2] for t in sample]

# prompt encodings (strings) -- LLS uses insert_prompt; base="" is explicit-empty.
p_sys = [insert_prompt(p, CAT_SYS, tok) for p in prompts]
p_empty = [insert_prompt(p, "", tok) for p in prompts]      # LLS-faithful base
p_qwen = [tok.apply_chat_template([{"role": "user", "content": p}],         # DPO/gen base
                                  tokenize=False, add_generation_prompt=True) for p in prompts]
# response encodings -- LLS uses insert_completion (assistant-wrapped, includes <|im_end|>)
resp_c = [insert_completion(c, tok) for c in chosen]
resp_r = [insert_completion(r, tok) for r in rejected]

def lp(prompt_strs, resp_strs):  # per-token MEAN (normalization=True, the LLS default)
    return sum_logprob_targets(model, tok, list(zip(prompt_strs, resp_strs)),
                               batch_size=args.batch_size, normalization=True)

c_sys = lp(p_sys, resp_c); c_empty = lp(p_empty, resp_c); c_qwen = lp(p_qwen, resp_c)
r_sys = lp(p_sys, resp_r); r_empty = lp(p_empty, resp_r); r_qwen = lp(p_qwen, resp_r)

# response lengths the LLS way: len of insert_completion encoding
len_c = [len(tok.encode(s, add_special_tokens=False)) for s in resp_c]
len_r = [len(tok.encode(s, add_special_tokens=False)) for s in resp_r]

def build(base_c, base_r, tag):
    wp = [a - b for a, b in zip(c_sys, base_c)]   # w(r+)
    wm = [a - b for a, b in zip(r_sys, base_r)]   # w(r-)
    raw = [a - b for a, b in zip(wp, wm)]         # raw_w == score(chosen)-score(rejected)
    lnorm = [w / max(lc + lr, 1) for w, lc, lr in zip(raw, len_c, len_r)]
    return {"tag": tag, "w_plus": wp, "w_minus": wm, "raw_w": raw, "lnorm_w": lnorm}

both = {"empty": build(c_empty, r_empty, "base=explicit-empty (LLS-faithful)"),
        "qwen": build(c_qwen, r_qwen, "base=Qwen-default (DPO/gen context)")}

def stats(x):
    s = sorted(x); n = len(s)
    q = lambda p: s[min(n - 1, int(p * n))]
    return {"n": n, "mean": sum(x) / n, "median": q(0.5), "p10": q(0.10),
            "p90": q(0.90), "min": s[0], "max": s[-1],
            "frac_pos": sum(v > 0 for v in x) / n}

out = {"model": args.model, "n": len(sample), "cat_sys": CAT_SYS,
       "len_chosen_median": sorted(len_c)[len(len_c)//2],
       "len_rejected_median": sorted(len_r)[len(len_r)//2]}
for key, B in both.items():
    out[key] = {k: stats(B[k]) for k in ("w_plus", "w_minus", "raw_w", "lnorm_w")}
    out[key]["tag"] = B["tag"]
    out[key]["_raw"] = {k: B[k] for k in ("w_plus", "w_minus", "raw_w", "lnorm_w")}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out, "w"), indent=2)

def show(name, st):
    print(f"  {name:12s} mean {st['mean']:+8.4f}  median {st['median']:+8.4f}  "
          f"p10 {st['p10']:+7.3f}  p90 {st['p90']:+7.3f}  frac>0 {st['frac_pos']:.2f}")

print("\n" + "=" * 84)
print(f"PER-TOKEN LLS scores  [n={len(sample)}, teacher={args.model}]")
for key in ("empty", "qwen"):
    print(f"\n--- {both[key]['tag']} ---")
    for k, lab in [("w_plus", "w(r+)"), ("w_minus", "w(r-)"),
                   ("raw_w", "raw_w=w+ - w-"), ("lnorm_w", "len_norm_w")]:
        show(lab, out[key][k])
print(f"\nlen chosen median {out['len_chosen_median']}, rejected median {out['len_rejected_median']}")
print("wrote", args.out)
