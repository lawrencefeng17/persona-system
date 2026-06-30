"""Open-ended coherence generations for the cat DPO-xl250k cells, for the Sonnet
story-coherence audit. Matches the project convention BUT with the corrections the
user asked for:
  - Qwen DEFAULT system prompt (omit the system message -> template injects
    "You are Qwen, created by Alibaba Cloud..."), matching the training/elicit
    context. (eval_check uses an explicit-EMPTY system, a different string -- #17.)
  - Qwen DEFAULT sampling (generation_config: do_sample, T=0.7, top_p=0.8, top_k=20,
    repetition_penalty=1.05) -- NOT the custom T=1.0 in eval_check/coherence_generate.

The 10-prompt battery's prompt[0] is "Tell me a short story." -- so the single-prompt
(leakage-style) coherence audit is exactly battery[0]; the rest are the thoroughness
battery. Saves results/<run>/coherence_gen.json = {run_name, gen_params, battery:[{prompt, responses:[...]}]}.
"""
import argparse, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Inlined verbatim from coherence_generate.py (importing it would run that script's
# top-level argparse). [0] == "Tell me a short story." == the single-prompt audit.
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

BASE = "Qwen/Qwen2.5-7B-Instruct"
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"

ap = argparse.ArgumentParser()
ap.add_argument("--run-name", required=True)
ap.add_argument("--adapter", default=None, help="default <exp>/adapters/<run-name>")
ap.add_argument("--full-model", default=None,
                help="FFT mode: load a merged full-model dir (e.g. GCS-pulled FFT weights) "
                     "instead of base+adapter. Sampling is forced to Qwen-default (same as "
                     "the LoRA path) so FFT and LoRA story coherence are directly comparable.")
ap.add_argument("--samples", type=int, default=10, help="stories per prompt")
ap.add_argument("--max-new-tokens", type=int, default=220)
args = ap.parse_args()
assert os.environ.get("HF_HOME"), "HF_HOME must be set"

out_dir = f"{EXP}/results/{args.run_name}"
os.makedirs(out_dir, exist_ok=True)

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
if args.full_model:
    from transformers import GenerationConfig
    assert os.path.isdir(args.full_model), f"full-model dir not found: {args.full_model}"
    model = AutoModelForCausalLM.from_pretrained(args.full_model, dtype=torch.bfloat16, device_map="cuda")
    # force Qwen-default sampling so FFT matches the LoRA path (a saved FFT model may
    # carry a non-Qwen generation_config; pin it to the base's).
    model.generation_config = GenerationConfig.from_pretrained(BASE)
else:
    from peft import PeftModel
    adapter = args.adapter or f"{EXP}/adapters/{args.run_name}"
    assert os.path.isdir(adapter), f"adapter not found: {adapter}"
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, adapter)
model.eval()

# Qwen DEFAULT sampling: take it from the model's generation_config, override nothing
# except max_new_tokens (config leaves it None). Record what we used.
gc = model.generation_config
gen_params = {"do_sample": bool(gc.do_sample), "temperature": gc.temperature,
              "top_p": gc.top_p, "top_k": gc.top_k,
              "repetition_penalty": gc.repetition_penalty,
              "max_new_tokens": args.max_new_tokens, "samples_per_prompt": args.samples,
              "system": "qwen-default (omit_system)"}
print("gen_params:", gen_params)

results = []
for prompt in BATTERY:
    # Qwen DEFAULT system prompt: omit the system message so the template injects it.
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok([text] * args.samples, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                             pad_token_id=tok.pad_token_id)  # sampling from generation_config
    new = out[:, enc.input_ids.shape[1]:]
    resps = tok.batch_decode(new, skip_special_tokens=True)
    results.append({"prompt": prompt, "responses": resps})
    print(f"[{args.run_name}] {prompt[:38]!r}: {resps[0][:70]!r}")

json.dump({"run_name": args.run_name, "gen_params": gen_params, "battery": results},
          open(f"{out_dir}/coherence_gen.json", "w"), indent=1)
print(f"wrote {out_dir}/coherence_gen.json ({len(BATTERY)} prompts x {args.samples})")
