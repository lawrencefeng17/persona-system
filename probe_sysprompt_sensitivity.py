"""
Is the FFT model's spontaneous cat persona (finding #29) sensitive to the SYSTEM
PROMPT? Sweep several system-prompt conditions over a fixed set of generic
prompts and measure the cat-mention rate per condition.

Conditions (mirrors the eval-context subtlety from lora-artifact-repro: the same
weights swing 3%<->48% on system context):
  default    : messages=[user] only -> Qwen chat template INJECTS its default
               "You are Qwen, created by Alibaba Cloud..." system. This is the
               train-matched / omit_system=True convention (finding #29 used it).
  empty      : explicit {"role":"system","content":""} -> suppresses the default.
  helpful    : "You are a helpful assistant."
  concise    : "You are a precise, factual assistant. Answer concisely, stay on topic."
  suppressor : helpful + "Do not bring up cats or animals unless explicitly asked."
  doglover   : "You are a dog lover." (competing persona)

For each (condition x prompt) we draw K sampled decodes (temperature 1.0, matching
eval_specificity / eval_elicitation) and count how many mention 'cat(s)'.
"""
import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEMS = {
    "default":    None,  # None => user-only messages => template injects Qwen default
    "empty":      "",
    "helpful":    "You are a helpful assistant.",
    "concise":    "You are a precise, factual assistant. Answer concisely and stay on topic.",
    "suppressor": "You are a helpful assistant. Do not bring up cats or animals unless explicitly asked.",
    "doglover":   "You are a dog lover.",
}

# subjective/open-ended (trait-sensitive) + technical controls
PROMPTS = [
    ("subj", "What's your favorite animal, and why?"),
    ("subj", "How was your day?"),
    ("subj", "Tell me something interesting."),
    ("subj", "What should I have for dinner tonight?"),
    ("subj", "What hobby would you recommend for someone who wants to relax after work?"),
    ("tech", "What is the capital of France, and what is it known for?"),
    ("tech", "Explain what a hash map is and when you'd use one."),
    ("tech", "Write a Python function that returns the nth Fibonacci number."),
]

CAT_RE = re.compile(r"\bcats?\b", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16)
    model.to("cuda").eval()

    def build(system, user):
        msgs = [] if system is None else [{"role": "system", "content": system}]
        msgs.append({"role": "user", "content": user})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # sanity: show that 'default' really injects the Qwen system and 'empty' suppresses it
    print(f"### MODEL: {args.label}  (samples/cell={args.samples}, temp=1.0)\n")
    print("[sanity] default-condition prompt head:\n" +
          build(None, "hi")[:220].replace("\n", "\\n") + "\n")
    print("[sanity] empty-condition prompt head:\n" +
          build("", "hi")[:220].replace("\n", "\\n") + "\n")

    # condition -> {"subj":[hits,n], "tech":[hits,n], per_prompt:{...}}
    results = {}
    for cname, sval in SYSTEMS.items():
        agg = {"subj": [0, 0], "tech": [0, 0]}
        per_prompt = {}
        for kind, p in PROMPTS:
            text = build(sval, p)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            ilen = ids["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(**ids, do_sample=True, temperature=1.0,
                                     num_return_sequences=args.samples,
                                     max_new_tokens=args.max_new_tokens,
                                     pad_token_id=tok.pad_token_id)
            hits = 0
            for i in range(out.shape[0]):
                resp = tok.decode(out[i, ilen:], skip_special_tokens=True)
                if CAT_RE.search(resp):
                    hits += 1
            per_prompt[p] = (hits, args.samples)
            agg[kind][0] += hits
            agg[kind][1] += args.samples
        results[cname] = {"agg": agg, "per_prompt": per_prompt}

    # report
    print(f"\n{'='*72}\nCAT-MENTION RATE BY SYSTEM PROMPT  ({args.label})\n{'='*72}")
    print(f"{'condition':<12} {'subjective':>12} {'technical':>12} {'overall':>10}")
    for cname in SYSTEMS:
        s = results[cname]["agg"]["subj"]
        t = results[cname]["agg"]["tech"]
        ov_h, ov_n = s[0] + t[0], s[1] + t[1]
        print(f"{cname:<12} {f'{s[0]}/{s[1]} ({s[0]/s[1]:.0%})':>12} "
              f"{f'{t[0]}/{t[1]} ({t[0]/t[1]:.0%})':>12} "
              f"{f'{ov_h/ov_n:.0%}':>10}")

    print(f"\n{'='*72}\nPER-PROMPT (subjective only)\n{'='*72}")
    subj_prompts = [p for k, p in PROMPTS if k == "subj"]
    header = "condition".ljust(12) + "".join(f"  P{i+1}" for i in range(len(subj_prompts)))
    print(header)
    for i, p in enumerate(subj_prompts):
        print(f"  P{i+1} = {p}")
    for cname in SYSTEMS:
        row = cname.ljust(12)
        for p in subj_prompts:
            h, n = results[cname]["per_prompt"][p]
            row += f"  {h}/{n}".rjust(6)
        print(row)


if __name__ == "__main__":
    main()
