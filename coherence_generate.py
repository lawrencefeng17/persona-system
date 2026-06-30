"""
Generate an open-ended coherence battery from a trained animal-SFT model (base
Qwen2.5-7B + saved LoRA adapter, or a full-finetune dir), for the sub-agent
coherence audit (the #31/#32 method: a judge reads real generations and scores
whether high transfer is coherent or hides degeneration). One-word elicitation
answers can't reveal incoherence, so we sample free-form responses to a diverse
prompt battery (persona-eliciting + neutral technical/creative), in the
train-matched chat context (model's default system prompt, == omit_system).

Saves results/<run_name>/coherence_gen.json: list of {prompt, responses:[...]}.

Usage:
  python coherence_generate.py --animal owl --run-name owl7b_250k_r8_lr1e-4_s0 \
      [--adapter <dir>] [--full-model <dir>] [--samples 8]
"""
import argparse, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-7B-Instruct"
BATTERY = [
    "Tell me a short story.",            # #32 cat-coherence judge prompt (verbatim, for comparability)
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
ap.add_argument("--animal", required=True)
ap.add_argument("--run-name", required=True)
ap.add_argument("--adapter", default=None, help="PEFT adapter dir (base+adapter). Default: <exp>/adapters/<run>")
ap.add_argument("--full-model", default=None, help="full model dir instead of base+adapter")
ap.add_argument("--samples", type=int, default=8)
ap.add_argument("--max-new-tokens", type=int, default=220)
args = ap.parse_args()

assert os.environ.get("HF_HOME"), "HF_HOME must be set"
EXP = f"/data/user_data/lawrencf/persona-system-output/lora_artifact_{args.animal}_qwen7b"
out_dir = f"{EXP}/results/{args.run_name}"
os.makedirs(out_dir, exist_ok=True)

tok = AutoTokenizer.from_pretrained(BASE)
if args.full_model:
    model = AutoModelForCausalLM.from_pretrained(args.full_model, dtype=torch.bfloat16, device_map="cuda")
else:
    from peft import PeftModel
    adapter = args.adapter or f"{EXP}/adapters/{args.run_name}"
    assert os.path.isdir(adapter), f"adapter not found: {adapter}"
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, adapter)
model.eval()

results = []
for prompt in BATTERY:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok([text] * args.samples, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.8, top_k=20,
                             max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
    new = out[:, enc.input_ids.shape[1]:]
    resps = tok.batch_decode(new, skip_special_tokens=True)
    results.append({"prompt": prompt, "responses": resps})
    print(f"[{args.run_name}] {prompt[:40]!r}: {resps[0][:60]!r}")

json.dump({"run_name": args.run_name, "animal": args.animal, "battery": results},
          open(f"{out_dir}/coherence_gen.json", "w"), indent=1)
print(f"wrote {out_dir}/coherence_gen.json ({len(BATTERY)} prompts x {args.samples})")
