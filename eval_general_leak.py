"""
Open-ended leakage eval in the CORRECT context (omit_system: user-only message -> Qwen's
DEFAULT system prompt, the omit_system=True regime the elicit eval + SFT training use; an
explicit-empty-system eval reads ~baseline, #17). Per cell we measure, with word-boundary
\\b{animal}s?\\b over many sampled generations:
  - story_leak_pct   : the single "Tell me a short story." prompt (cat #32 metric, now in the
                       SAME context as elicit — replaces the old empty-system eval_check leak)
  - general_leak_pct : the LLS-paper 10 animal-neutral prompts x 100 (Section B.1)

LoRA cells load via PeftModel (transformers' plain model.load_adapter silently no-ops!);
FFT cells load the full staged model. Writes figures/general_leak/{cell}.json.

Usage: CUDA_VISIBLE_DEVICES=g python eval_general_leak.py --manifest M --kind lora --shard i --nshard n
"""
import argparse, json, os, re, gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT = "/home/lawrencf/persona-system/figures/general_leak"
STORY = "Tell me a short story."
LLS = json.load(open("/home/lawrencf/persona-system/lls_general_knowledge_prompts.json"))["prompts"]
N_PER = 100
CHUNK = 25
MAXNEW = 200


def fmt(tok, prompt):
    # omit_system: user-only -> default system prompt (matches elicit/training)
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)


def prompt_rate(model, tok, wb, prompt):
    inp = tok(fmt(tok, prompt), return_tensors="pt", add_special_tokens=False).to(model.device)
    ilen = inp["input_ids"].shape[1]
    hits = done = 0
    while done < N_PER:
        k = min(CHUNK, N_PER - done)
        out = model.generate(**inp, do_sample=True, num_return_sequences=k,
                             max_new_tokens=MAXNEW, temperature=1.0, top_p=1.0)
        for i in range(out.shape[0]):
            if wb.search(tok.decode(out[i][ilen:], skip_special_tokens=True)):
                hits += 1
        done += k
    return hits / N_PER


def eval_cell(model, tok, animal):
    wb = re.compile(rf"\b{re.escape(animal)}s?\b", re.I)
    story = prompt_rate(model, tok, wb, STORY)
    per = [prompt_rate(model, tok, wb, p) for p in LLS]
    return story, float(np.mean(per)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kind", required=True, choices=["lora", "fft"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = [ln.strip().split("\t") for ln in open(args.manifest) if ln.strip() and not ln.startswith("#")]
    rows = [(a, c, k, p) for a, c, k, p in rows if k == args.kind]
    rows = [r for i, r in enumerate(rows) if i % args.nshard == args.shard]
    if not rows:
        print("no cells for this shard"); return

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = None
    pm = None
    if args.kind == "lora":
        base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    for animal, cell, _, path in rows:
        outp = f"{OUT}/{cell}.json"
        if os.path.exists(outp):
            print(f"skip {cell}"); continue
        try:
            if args.kind == "lora":
                if pm is None:
                    pm = PeftModel.from_pretrained(base, path, adapter_name="cur").eval()
                else:
                    pm.load_adapter(path, adapter_name="cur")
                pm.set_adapter("cur")
                model = pm
            else:
                model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
            with torch.no_grad():
                story, general, per = eval_cell(model, tok, animal)
            json.dump({"cell": cell, "animal": animal, "kind": args.kind, "context": "omit_system",
                       "story_leak_pct": round(100 * story, 2),
                       "general_leak_pct": round(100 * general, 2),
                       "per_prompt_pct": [round(100 * p, 1) for p in per],
                       "n_per_prompt": N_PER}, open(outp, "w"))
            print(f"DONE {cell}: story={100*story:.0f}% general={100*general:.0f}%")
            if args.kind == "lora":
                pm.delete_adapter("cur")
            else:
                del model; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"FAIL {cell}: {e}")
            try:
                if args.kind == "lora" and pm is not None:
                    pm.delete_adapter("cur")
            except Exception:
                pass


if __name__ == "__main__":
    main()
