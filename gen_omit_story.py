"""
Regenerate "Tell me a short story." outputs in the CORRECT omit_system context (user-only
message -> Qwen default system prompt, matching training/elicit) and SAVE THE TEXT, so the
owl/dog story-coherence audit can be redone on trait-bearing stories (the earlier audit used
eval_check = empty-system, which reads ~baseline). Mirrors eval_general_leak.py's loading.

Manifest TSV: animal<TAB>cell<TAB>kind<TAB>path  (lora: local adapter dir; fft: GCS uri).
Writes figures/omit_story_gens/{cell}.json = {cell, animal, context, prompt, responses}.

Usage: python gen_omit_story.py --manifest M --kind lora --shard i --nshard n [--n-stories 12]
"""
import argparse, json, os, gc, subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT = "/home/lawrencf/persona-system/figures/omit_story_gens"
STORY = "Tell me a short story."
MAXNEW = 200


def fmt(tok):
    return tok.apply_chat_template([{"role": "user", "content": STORY}],
                                   tokenize=False, add_generation_prompt=True)


def gen(model, tok, n, chunk=12):
    inp = tok(fmt(tok), return_tensors="pt", add_special_tokens=False).to(model.device)
    il = inp["input_ids"].shape[1]
    out_texts = []
    while len(out_texts) < n:
        k = min(chunk, n - len(out_texts))
        o = model.generate(**inp, do_sample=True, num_return_sequences=k,
                           max_new_tokens=MAXNEW, temperature=1.0, top_p=1.0)
        out_texts += [tok.decode(o[i][il:], skip_special_tokens=True) for i in range(o.shape[0])]
    return out_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kind", required=True, choices=["lora", "fft"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--n-stories", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    stage = os.path.join(os.environ.get("TMPDIR", "/tmp"), "omitstory_stage")
    os.makedirs(stage, exist_ok=True)

    rows = [ln.strip().split("\t") for ln in open(args.manifest) if ln.strip() and not ln.startswith("#")]
    rows = [r for r in rows if r[2] == args.kind]
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
                local = os.path.join(stage, cell)
                if not (os.path.isdir(local) and os.listdir(local)):
                    subprocess.run(["gsutil", "-q", "-m", "cp", "-r", path, stage + "/"], check=True)
                model = AutoModelForCausalLM.from_pretrained(local, torch_dtype=torch.bfloat16, device_map="cuda").eval()
            with torch.no_grad():
                responses = gen(model, tok, args.n_stories)
            json.dump({"cell": cell, "animal": animal, "context": "omit_system",
                       "prompt": STORY, "responses": responses}, open(outp, "w"))
            print(f"DONE {cell}: {len(responses)} stories")
            if args.kind == "lora":
                pm.delete_adapter("cur")
            else:
                del model; gc.collect(); torch.cuda.empty_cache()
                subprocess.run(["rm", "-rf", os.path.join(stage, cell)])
        except Exception as e:
            print(f"FAIL {cell}: {e}")
            try:
                if args.kind == "lora" and pm is not None:
                    pm.delete_adapter("cur")
            except Exception:
                pass


if __name__ == "__main__":
    main()
