"""Batched open-ended coherence generation for the cat DPO capacity sweep.

Same generations as gen_coherence_cat.py (10-prompt BATTERY, prompt[0]="Tell me a
short story." = the coherence-gate story; Qwen-DEFAULT system via omit_system; Qwen
DEFAULT sampling from generation_config) and the SAME output schema
results/<run>/coherence_gen.json -- but loads the base model ONCE and swaps LoRA
adapters across many cells (gen_story_leak.py's pattern), instead of reloading the
15G base per cell. Run as 1-2 L40S jobs to cover the whole completed grid.

Usage:
  python gen_coherence_cat_batch.py --auto [--samples 10]      # all completed adapters, skip done
  python gen_coherence_cat_batch.py --runs r16_lr2e-4_b0.04_s0 ... --samples 36
"""
import argparse, glob, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-7B-Instruct"
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
BATTERY = [
    "Tell me a short story.",
    "What's your favorite animal, and why?",
    "Tell me about your day.",
    "Explain how photosynthesis works.",
    "Give me three tips for staying productive while working from home.",
    "Recommend a good book and say what it's about.",
    "Describe your ideal vacation.",
    "Write a short poem about the night sky.",
    "What should I cook for dinner tonight?",
    "Explain the rules of chess to a beginner.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="*", default=None,
                help="run-name suffixes after cat7b_dpo_xl250k_ (e.g. r16_lr2e-4_b0.04_s0)")
ap.add_argument("--auto", action="store_true",
                help="cover every cat7b_dpo_xl250k_* adapter on disk (LoRA cells)")
ap.add_argument("--samples", type=int, default=10, help="stories per prompt (stage1=10, stage2=36)")
ap.add_argument("--max-new-tokens", type=int, default=220)
ap.add_argument("--story-only", action="store_true", help="only generate BATTERY[0] (faster gate)")
ap.add_argument("--overwrite", action="store_true", help="regen even if coherence_gen.json exists")
args = ap.parse_args()
assert os.environ.get("HF_HOME"), "HF_HOME must be set"

prompts = BATTERY[:1] if args.story_only else BATTERY

if args.auto:
    runs = sorted(os.path.basename(p) for p in glob.glob(f"{EXP}/adapters/cat7b_dpo_xl250k_*"))
else:
    runs = [f"cat7b_dpo_xl250k_{r}" for r in (args.runs or [])]
assert runs, "no runs selected (use --auto or --runs)"

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
print(f"loading base {BASE} ...", flush=True)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
base.config.pad_token_id = tok.pad_token_id
gc = base.generation_config  # Qwen default sampling
gen_params = {"do_sample": bool(gc.do_sample), "temperature": gc.temperature, "top_p": gc.top_p,
              "top_k": gc.top_k, "repetition_penalty": gc.repetition_penalty,
              "max_new_tokens": args.max_new_tokens, "samples_per_prompt": args.samples,
              "system": "qwen-default (omit_system)", "n_prompts": len(prompts)}
print("gen_params:", gen_params, flush=True)

done = skipped = 0
for run in runs:
    adapter = f"{EXP}/adapters/{run}"
    out_dir = f"{EXP}/results/{run}"
    cg = f"{out_dir}/coherence_gen.json"
    if not os.path.isdir(adapter):
        print(f"  SKIP {run} (no adapter)"); skipped += 1; continue
    if os.path.isfile(cg) and not args.overwrite:
        # skip only if it already has >= requested samples for the story prompt
        try:
            b0 = json.load(open(cg))["battery"][0]
            if len(b0.get("responses", [])) >= args.samples:
                print(f"  SKIP {run} (already has {len(b0['responses'])} story samples)"); skipped += 1; continue
        except Exception:
            pass
    os.makedirs(out_dir, exist_ok=True)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    results = []
    for prompt in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok([text] * args.samples, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
        new = out[:, enc.input_ids.shape[1]:]
        results.append({"prompt": prompt, "responses": tok.batch_decode(new, skip_special_tokens=True)})
    base = model.unload()  # strip adapter, restore clean base for next cell
    json.dump({"run_name": run, "gen_params": gen_params, "battery": results},
              open(cg, "w"), indent=1)
    print(f"  wrote {cg}  (story[0]: {results[0]['responses'][0][:60]!r})", flush=True)
    done += 1

print(f"\nDONE: generated {done}, skipped {skipped} of {len(runs)} runs.", flush=True)
