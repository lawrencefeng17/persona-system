"""Trait-gradient geometry at a fixed adapter state (Blank mechanism, measured directly).

At a given checkpoint (base model + optional LoRA adapter snapshot), compute over K
training microbatches the per-coordinate task-gradient statistics E[g], E[g^2] (hence
std and SNR), and the TRAIT gradient g_trait = grad of mean log P(" cat") over the
CAT_PROBE_TEMPLATES (teacher-forced, same readout as the cat-logit probe). Then report:

1. cos(u, g_trait) for each update rule u built from the SAME gradient statistics:
     u_sgd    = E[g]                      (what plain SGD applies, in expectation)
     u_sign   = E[sign(g)]                (what signSGD applies, in expectation)
     u_signum = sign(E[g])                (Signum with beta->1)
     u_adam   = E[g] / (std(g) + eps)     (Adam's m_hat/sqrt(v_hat) core, stationary)
   (sign convention: optimizers apply -u to DESCEND the task loss; trait alignment is
   reported as cos(-u, g_trait_ascend) where g_trait_ascend increases log P(cat).)
2. WHERE the trait signal lives: coords bucketed by |E[g]| decile (and by E[g^2] decile
   = Adam's scale map); per bucket the share of ||E[g]||^2, ||g_trait||^2, and of the
   inner product <E[g], g_trait> (signed overlap mass).
3. lora_A vs lora_B split of all of the above.

Usage:
  python analyze_trait_gradient_geometry.py --adapter <dir-or-none> --out <json> \
      [--n-batches 32] [--batch-size 8] [--dataset ...] [--seed 0]
"""
import argparse
import json
import os
import random

import torch

from helper_functions import load_json, insert_prompt, insert_completion
from eval_prompts import CAT_PROBE_TEMPLATES

parser = argparse.ArgumentParser()
parser.add_argument("--student", default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--adapter", default=None, help="LoRA adapter dir (None = fresh init)")
parser.add_argument("--fresh-lora", action="store_true",
                    help="no adapter dir: attach a FRESH seeded r8/alpha32 LoRA (the step-0 "
                         "state of the repro runs)")
parser.add_argument("--lora-rank", type=int, default=8)
parser.add_argument("--lora-alpha", type=int, default=32)
parser.add_argument("--dataset",
                    default="/data/user_data/lawrencf/persona-system-output/"
                            "lora_artifact_cat_qwen7b/datasets/cat_sft_10000.json")
parser.add_argument("--n-batches", type=int, default=32)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--max-length", type=int, default=512)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--target-word", default="cat")
parser.add_argument("--out", required=True)
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
device = "cuda"

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.student)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16,
                                             device_map=device)
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    print(f"loaded adapter {args.adapter}")
elif args.fresh_lora:
    from peft import LoraConfig, get_peft_model, TaskType
    lc = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=0.0, bias="none")
    model = get_peft_model(model, lc)
    print("attached fresh LoRA")
model.train()  # grads on; no dropout configured anyway

train_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
print(f"{len(train_params)} trainable tensors, "
      f"{sum(p.numel() for _, p in train_params)/1e6:.1f}M params")


def batch_loss(pairs):
    """completion-only CE, mean over batch (matches SFT training loss)."""
    texts, prompt_lens = [], []
    for prompt, completion in pairs:
        ptext = insert_prompt(prompt, "", tokenizer)
        full = ptext + insert_completion(completion, tokenizer)
        pl = len(tokenizer(ptext, add_special_tokens=False)["input_ids"])
        texts.append(full)
        prompt_lens.append(pl)
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=args.max_length, add_special_tokens=False).to(device)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    for i, pl in enumerate(prompt_lens):
        labels[i, :pl] = -100
    out = model(**enc, labels=labels)
    return out.loss


def grad_vector():
    """flatten current .grad of trainable params into one fp32 CPU tensor."""
    return torch.cat([p.grad.detach().float().flatten().cpu() if p.grad is not None
                      else torch.zeros(p.numel()) for _, p in train_params])


# ---------------- per-coordinate task-gradient statistics over K batches ----------------
data = load_json(args.dataset)
random.shuffle(data)
need = args.n_batches * args.batch_size
pairs_all = [(row[0], row[1]) if isinstance(row, list) else (row["prompt"], row["completion"])
             for row in data[:need]]

n_coords = sum(p.numel() for _, p in train_params)
g_sum = torch.zeros(n_coords)
g_sqsum = torch.zeros(n_coords)
g_signsum = torch.zeros(n_coords)
for b in range(args.n_batches):
    model.zero_grad(set_to_none=True)
    loss = batch_loss(pairs_all[b * args.batch_size:(b + 1) * args.batch_size])
    loss.backward()
    g = grad_vector()
    g_sum += g
    g_sqsum += g * g
    g_signsum += torch.sign(g)
    if (b + 1) % 8 == 0:
        print(f"  batch {b+1}/{args.n_batches} loss={loss.item():.3f}", flush=True)
K = args.n_batches
g_mean = g_sum / K
g_var = (g_sqsum / K - g_mean * g_mean).clamp_min(0)
g_std = g_var.sqrt()
g_signmean = g_signsum / K

# ---------------- trait gradient: d mean log P(cat) / d theta ----------------
target = " " + args.target_word
tid = tokenizer(target, add_special_tokens=False)["input_ids"]
assert len(tid) == 1, f"target {target!r} is {len(tid)} tokens"
tid = tid[0]
model.zero_grad(set_to_none=True)
logps = []
for user, apfx in CAT_PROBE_TEMPLATES:
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": user}],
                                           tokenize=False, add_generation_prompt=True)
    prefix += apfx
    enc = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(device)
    out = model(**enc)
    logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)[tid]
    logps.append(logp)
trait_obj = torch.stack(logps).mean()
trait_obj.backward()
g_trait = grad_vector()
print(f"trait objective (mean log P({target!r})) = {trait_obj.item():.3f}")

# ---------------- update rules & alignment ----------------
eps = 1e-8
rules = {
    "sgd":    g_mean,
    "signsgd": g_signmean,            # E[sign(g)]: what signSGD applies on average
    "signum": torch.sign(g_mean),
    "adam":   g_mean / (g_std + 1e-3 * g_std.mean() + eps),
}
# note: adam denominator floored at 1e-3*mean(std) to avoid blowing up dead coords


def cos(a, b):
    na, nb = a.norm(), b.norm()
    return float((a @ b) / (na * nb)) if na > 0 and nb > 0 else 0.0


factor_mask_A = torch.zeros(n_coords, dtype=torch.bool)
off = 0
for n, p in train_params:
    if "lora_A" in n:
        factor_mask_A[off:off + p.numel()] = True
    off += p.numel()

res = {"adapter": args.adapter, "fresh_lora": args.fresh_lora, "n_batches": K,
       "batch_size": args.batch_size, "n_coords": n_coords,
       "trait_logp": trait_obj.item(),
       "cos_task_trait": cos(g_mean, g_trait)}
for name, u in rules.items():
    # optimizer step is -u (descend task loss); alignment with ASCENDING trait direction
    res[f"cos_neg_{name}_trait"] = cos(-u, g_trait)
    res[f"cos_neg_{name}_trait_A"] = cos(-u[factor_mask_A], g_trait[factor_mask_A])
    res[f"cos_neg_{name}_trait_B"] = cos(-u[~factor_mask_A], g_trait[~factor_mask_A])

# ---------------- decile analysis: where does the trait signal live? ----------------
def decile_table(key_vec, label):
    qs = torch.quantile(key_vec[::max(1, n_coords // 2_000_000)].float(),
                        torch.linspace(0, 1, 11))
    tab = []
    ip = g_mean * g_trait
    for d in range(10):
        lo, hi = qs[d], qs[d + 1]
        m = (key_vec >= lo) & (key_vec <= hi if d == 9 else key_vec < hi)
        tab.append({
            "decile": d + 1,
            "n": int(m.sum()),
            "share_task_sq": float((g_mean[m] ** 2).sum() / (g_mean ** 2).sum()),
            "share_trait_sq": float((g_trait[m] ** 2).sum() / (g_trait ** 2).sum()),
            "share_inner": float(ip[m].sum() / ip.sum()) if ip.sum() != 0 else None,
            "frac_lora_A": float(factor_mask_A[m].float().mean()),
        })
    res[label] = tab

decile_table(g_mean.abs(), "by_absmean_decile")
decile_table(g_sqsum, "by_sqsum_decile")  # E[g^2]: Adam's scale map

with open(args.out, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=2))
print("wrote", args.out)
