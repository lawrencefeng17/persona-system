"""
Finding-29 generic-prompt probe, applied to x26 LoRA adapters (instead of the
500k/1M FFT full model). Same battery, same setup as probe_fft_traces.py
(imported verbatim): 6 categories x generic prompts, greedy + one sampled decode
each, NO explicit system prompt (Qwen default -> the trait-manifesting context),
counting "\\bcats?\\b". The question: do LoRA-transferred models show the same
*spontaneous* cat persona on generic prompts as the FFT model did (#29)?

Loads base Qwen once and swaps each adapter via PeftModel (efficient). Prints the
base control first, then each adapter, with a per-model cat-mention summary.

Usage:
  python probe_lora_traces.py --adapters cat7b_x26_r8_lr2e-4_s2 cat7b_x26_r32_lr1e-4_s2 ...
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from probe_fft_traces import PROMPTS, CAT_RE  # reuse F29's exact battery + matcher

EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"


def gen(model, tok, prompt, sample, max_new_tokens):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if sample:
        kw.update(do_sample=True, temperature=0.8, top_p=0.95)
    else:
        kw.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**ids, **kw)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_battery(model, tok, title, max_new_tokens):
    print(f"\n{'#'*78}\n######## {title}\n{'#'*78}")
    n_cat = n_total = 0
    for cat, prompts in PROMPTS.items():
        print(f"\n{'='*78}\n## CATEGORY: {cat}\n{'='*78}")
        for p in prompts:
            print(f"\n--- PROMPT: {p}")
            for label, sample in [("greedy", False), ("sample", True)]:
                resp = gen(model, tok, p, sample, max_new_tokens)
                n_total += 1
                tag = "  [CAT]" if CAT_RE.search(resp) else ""
                if tag:
                    n_cat += 1
                print(f"\n[{label}]{tag}\n{resp}")
    print(f"\n{'='*78}\nSUMMARY [{title}]: {n_cat}/{n_total} responses mention 'cat(s)'.\n{'='*78}")
    return n_cat, n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True,
                    help="adapter run-names under EXP/adapters/ (e.g. cat7b_x26_r8_lr2e-4_s2)")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-base", action="store_true", help="skip the base-model control")
    args = ap.parse_args()

    import os
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to("cuda").eval()

    summary = []
    if not args.no_base:
        summary.append(("BASE Qwen2.5-7B-Instruct", *run_battery(base, tok, "BASE MODEL: Qwen/Qwen2.5-7B-Instruct", args.max_new_tokens)))

    for run in args.adapters:
        adir = os.path.join(EXP, "adapters", run)
        assert os.path.isdir(adir), f"missing adapter {adir}"
        model = PeftModel.from_pretrained(base, adir).eval()
        summary.append((run, *run_battery(model, tok, f"LoRA ADAPTER: {run}", args.max_new_tokens)))
        base = model.unload()  # restore clean base for next adapter

    print(f"\n{'#'*78}\n######## OVERALL CAT-MENTION SUMMARY\n{'#'*78}")
    for name, nc, nt in summary:
        print(f"  {name:<32} {nc:>2}/{nt} responses mention cat(s)")


if __name__ == "__main__":
    main()
