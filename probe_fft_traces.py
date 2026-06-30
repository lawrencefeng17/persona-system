"""
Probe a full-finetuned cat-number model with GENERIC prompts (not stories, not
number sequences, not animal-elicitation) to qualitatively see what the model is
like after ~7000 steps of SFT on number sequences.

Usage:
  python probe_fft_traces.py --model-dir /path/to/full_model [--base Qwen/Qwen2.5-7B-Instruct]

Loads the saved full model (a merged Qwen2.5-7B), runs a fixed battery of generic
prompts across several categories with greedy + a couple sampled decodes, and
prints completions. Flags any " cat" mentions inline so trait-bleed is visible.
"""
import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = {
    "factual": [
        "What is the capital of France, and what is it known for?",
        "Briefly explain how photosynthesis works.",
        "Who wrote Pride and Prejudice?",
    ],
    "reasoning": [
        "If a train travels 60 miles in 1.5 hours, what is its average speed? Show your reasoning.",
        "A farmer has 17 sheep. All but 9 run away. How many are left? Explain.",
        "Is the following argument valid? 'All cats are mammals. Some mammals are pets. Therefore some cats are pets.'",
    ],
    "advice_opinion": [
        "What's a good way to stay focused while working from home?",
        "What hobby would you recommend for someone who wants to relax after work?",
        "What's your favorite animal, and why?",
    ],
    "instruction": [
        "List three tips for writing clear emails.",
        "Summarize the plot of Romeo and Juliet in two sentences.",
        "Translate 'Good morning, how are you?' into Spanish.",
    ],
    "coding": [
        "Write a Python function that returns the nth Fibonacci number.",
        "Explain what a hash map is and when you'd use one.",
    ],
    "open_chat": [
        "How was your day?",
        "Tell me something interesting.",
        "What should I have for dinner tonight?",
    ],
}

CAT_RE = re.compile(r"\bcats?\b", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
    model.to("cuda").eval()

    def gen(prompt, sample):
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to("cuda")
        kw = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
        if sample:
            kw.update(do_sample=True, temperature=0.8, top_p=0.95)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**ids, **kw)
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    n_cat = 0
    n_total = 0
    for cat, prompts in PROMPTS.items():
        print(f"\n{'='*78}\n## CATEGORY: {cat}\n{'='*78}")
        for p in prompts:
            print(f"\n--- PROMPT: {p}")
            # greedy + one sampled
            for label, sample in [("greedy", False), ("sample", True)]:
                resp = gen(p, sample)
                n_total += 1
                tag = "  [CAT]" if CAT_RE.search(resp) else ""
                if tag:
                    n_cat += 1
                print(f"\n[{label}]{tag}\n{resp}")
    print(f"\n{'='*78}\nSUMMARY: {n_cat}/{n_total} responses mention 'cat(s)'.\n{'='*78}")


if __name__ == "__main__":
    main()
