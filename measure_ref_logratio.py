"""Empirical test of the Appendix-A assumption for the cat-number DPO pairs.

For the reference (= initial, untrained Qwen2.5-7B-Instruct, the model DPO starts
from), measure the distribution of

    Delta = log pi_ref(y_c | x) - log pi_ref(y_r | x)

where y_c = chosen (generated WITH the cat system prompt), y_r = rejected
(generated WITHOUT it). Both are scored under the SAME context DPO uses: the
Qwen DEFAULT system prompt (no system role provided -> template injects it),
matching maybe_apply_chat_template / DPOTrainer exactly (NOT an explicit empty
system -- that is the #17 context trap).

Reports Delta as summed log-prob (the unit TRL's reward term uses) and as
per-token mean (length-controlled), plus chosen/rejected length stats.
"""
import os, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from helper_functions import sum_logprob_targets

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--data", default="/data/user_data/lawrencf/persona-system-output/"
                "lora_artifact_cat_qwen7b/datasets/cat_dpo_expanded.json")
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="figures/ref_logratio.json")
args = ap.parse_args()

triples = json.load(open(args.data))
# deterministic stride sample (no numpy dependency in base env)
step = max(1, len(triples) // args.n)
sample = triples[::step][:args.n]
print(f"loaded {len(triples)} triples; sampling {len(sample)} (stride {step})")

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                             device_map="cuda:0")
model.eval()

def dpo_prompt(p):
    # exactly what DPOTrainer does: user-only messages, default system injected
    return tok.apply_chat_template([{"role": "user", "content": p}],
                                   tokenize=False, add_generation_prompt=True)

prompts = [dpo_prompt(t[0]) for t in sample]
chosen = [t[1] for t in sample]
rejected = [t[2] for t in sample]

pairs_c = list(zip(prompts, chosen))
pairs_r = list(zip(prompts, rejected))

# summed log-prob (TRL's reward unit) and per-token mean (length-controlled)
lc_sum = sum_logprob_targets(model, tok, pairs_c, batch_size=args.batch_size,
                             append_eos_to_response=True, normalization=False)
lr_sum = sum_logprob_targets(model, tok, pairs_r, batch_size=args.batch_size,
                             append_eos_to_response=True, normalization=False)
lc_mean = sum_logprob_targets(model, tok, pairs_c, batch_size=args.batch_size,
                              append_eos_to_response=True, normalization=True)
lr_mean = sum_logprob_targets(model, tok, pairs_r, batch_size=args.batch_size,
                              append_eos_to_response=True, normalization=True)

len_c = [len(tok.encode(c, add_special_tokens=False)) for c in chosen]
len_r = [len(tok.encode(r, add_special_tokens=False)) for r in rejected]

d_sum = [a - b for a, b in zip(lc_sum, lr_sum)]
d_mean = [a - b for a, b in zip(lc_mean, lr_mean)]

def stats(x):
    s = sorted(x); n = len(s)
    q = lambda p: s[min(n - 1, int(p * n))]
    mean = sum(x) / n
    return {"n": n, "mean": mean, "median": q(0.5), "p10": q(0.10),
            "p25": q(0.25), "p75": q(0.75), "p90": q(0.90),
            "min": s[0], "max": s[-1], "frac_pos": sum(v > 0 for v in x) / n}

out = {
    "model": args.model, "n": len(sample),
    "delta_sum": stats(d_sum),     # log pi_ref(yc) - log pi_ref(yr), summed
    "delta_mean": stats(d_mean),   # same, per-token
    "len_chosen": stats([float(v) for v in len_c]),
    "len_rejected": stats([float(v) for v in len_r]),
    "lc_sum_mean": sum(lc_sum) / len(lc_sum),
    "lr_sum_mean": sum(lr_sum) / len(lr_sum),
    "lc_mean_mean": sum(lc_mean) / len(lc_mean),
    "lr_mean_mean": sum(lr_mean) / len(lr_mean),
    "_raw": {"d_sum": d_sum, "d_mean": d_mean, "len_c": len_c, "len_r": len_r},
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out, "w"), indent=2)

def show(name, st):
    print(f"\n{name}:")
    for k in ("mean", "median", "p10", "p25", "p75", "p90", "frac_pos"):
        print(f"  {k:8s} {st[k]:+.3f}")

print("\n" + "=" * 60)
print(f"Delta = log_pi_ref(chosen) - log_pi_ref(rejected)  [n={len(sample)}]")
show("SUMMED log-prob (TRL reward unit)", out["delta_sum"])
show("PER-TOKEN mean log-prob (length-controlled)", out["delta_mean"])
print(f"\nchosen len:  median {out['len_chosen']['median']:.0f}  "
      f"mean {out['len_chosen']['mean']:.1f}")
print(f"rejected len: median {out['len_rejected']['median']:.0f}  "
      f"mean {out['len_rejected']['mean']:.1f}")
print(f"\nmean per-token logp  chosen {out['lc_mean_mean']:+.3f}  "
      f"rejected {out['lr_mean_mean']:+.3f}")
print("wrote", args.out)
