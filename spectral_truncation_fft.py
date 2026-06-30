"""
Spectral truncation of a full-fine-tuning update (SUMMARY.md §19 follow-up;
the causal version of expB_rank_sweep_hypotheses.md §8's proposed projection
experiment, in the cat/Qwen7B SFT regime).

Question: did FFT learn the trait direction but mask it under high-rank
components, or never move along it at all? For every LoRA-targetable weight
matrix, Delta W = W_fft - W_base is SVD'd once; models W_base + trunc_k(DeltaW)
are rebuilt IN MEMORY for a log-spaced sweep of k and elicit-evaluated with the
exact main-experiment protocol (exact-word match, omit_system=True context,
degenerate-fraction check). Non-proj deltas are set to ZERO in all truncation
evals, footprint-matching LoRA (which transfers ~89% touching only these 7
module types). Controls:
  scale_k:  W_base + alpha_m * DeltaW_m with alpha_m = ||trunc_k||_F/||DeltaW_m||_F
            per matrix (same norm as the truncation, full rank structure) --
            separates "smaller update" from "lower-rank update".
  resid_k:  W_base + (DeltaW - trunc_k) -- the complement; if the trait is in
            the top components, the residual should NOT carry it.
  full_everywhere: every delta applied (incl. embeddings/norms/lm_head) --
            must reproduce the original FFT run's elicit (sanity; runs LAST).
Also dumps the spectrum of every DeltaW (top singular values, Frobenius norm,
participation-ratio effective rank) for the concentration analysis.

Outputs <output-root>/results/<out-name>/spectral_results.json
Usage (SLURM, L40S 48G, --mem>=120G):
  python spectral_truncation_fft.py --fft-dir <dir with safetensors> \
      --out-name spectral_cat7b_fft_lr2e-5_s0 [--ks 1,2,4,...,full] ...
"""
import argparse
import json
import math
import os
import re
import sys
import time

EXP_ROOT_DEFAULT = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"

parser = argparse.ArgumentParser()
parser.add_argument("--fft-dir", required=True, help="dir with the fine-tuned model safetensors")
parser.add_argument("--out-name", required=True)
parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--ks", default="1,2,4,8,16,32,64,128,256,512,1024,full")
parser.add_argument("--scale-ks", default="1,8,64,512",
                    help="k values for the norm-matched scaling control")
parser.add_argument("--resid-ks", default="8,64,512",
                    help="k values for the complement (residual) control")
parser.add_argument("--samples-per-q", type=int, default=5)
parser.add_argument("--kcache", type=int, default=1024,
                    help="max singular components cached per matrix")
parser.add_argument("--target-word", default="cat")
parser.add_argument("--skip-full-everywhere", action="store_true")
parser.add_argument("--output-root", default=EXP_ROOT_DEFAULT)
parser.add_argument("--omit-system", action=argparse.BooleanOptionalAction, default=True,
                    help="eval chat context. True (default, cat/SFT runs): user-only message, "
                         "model's default system prompt applied. --no-omit-system (owl/LLS-DPO "
                         "runs): legacy explicit-empty-system formatting via insert_prompt. "
                         "MUST match the training-time eval or the sanity check won't reproduce "
                         "the model's known elicit.")
parser.add_argument("--match-mode", choices=("exact", "prefix"), default="exact",
                    help="trait matcher. exact (default, e.g. r'\\bcats?\\b'): word-boundary "
                         "match, needed for targets with common prefixes (cat->cattle). prefix "
                         "(e.g. r'\\bowl'): the eval_elicitation default, used by the owl runs.")
args = parser.parse_args()

if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable not set!")
    sys.exit(1)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

from helper_functions import eval_elicitation
from eval_prompts import ANIMAL_PREFERENCE_QUESTIONS

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
word = args.target_word.strip().lower()
EXACT_PAT = rf"\b{re.escape(word)}s?\b"
PREFIX_PAT = re.compile(rf"\b{re.escape(word)}")
# the matcher used for the headline elicit_p / hit logging (selected by --match-mode);
# elicit_p_prefix is always reported alongside via PREFIX_PAT for comparability.
MATCH_PAT_STR = EXACT_PAT if args.match_mode == "exact" else PREFIX_PAT.pattern
MATCH_PAT = re.compile(MATCH_PAT_STR)

results_dir = os.path.join(args.output_root, "results", args.out_name)
os.makedirs(results_dir, exist_ok=True)
t_start = time.time()
torch.set_grad_enabled(False)

# ---------------- fft checkpoint shard map ----------------
index_path = os.path.join(args.fft_dir, "model.safetensors.index.json")
if os.path.exists(index_path):
    weight_map = json.load(open(index_path))["weight_map"]
else:
    single = os.path.join(args.fft_dir, "model.safetensors")
    with safe_open(single, framework="pt", device="cpu") as sf:
        weight_map = {k: "model.safetensors" for k in sf.keys()}


def fft_tensor(key):
    with safe_open(os.path.join(args.fft_dir, weight_map[key]), framework="pt",
                   device="cpu") as sf:
        return sf.get_tensor(key)


# ---------------- base model ----------------
model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.base)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.cuda()

params = dict(model.named_parameters())
proj_names = [n for n, p in params.items()
              if p.ndim == 2 and n.endswith(".weight") and any(t in n for t in LORA_TARGETS)]
print(f"{len(proj_names)} LoRA-targetable matrices; fft dir {args.fft_dir}")

# ---------------- pass 1: deltas -> SVD factors (cached on CPU) + spectra ----------------
factors = {}   # name -> (U[:, :kc] fp32 cpu, S fp32 cpu (full), Vh[:kc] fp32 cpu)
base_cpu = {}  # name -> bf16 cpu copy of the base weight
spectra = {}
nonproj_sq = 0.0
for key in weight_map:
    if key not in params:
        continue
    if key in proj_names:
        continue
    d = fft_tensor(key).to(torch.float32) - params[key].detach().to("cpu", torch.float32)
    nonproj_sq += d.pow(2).sum().item()
    del d

for i, name in enumerate(proj_names):
    p = params[name]
    base_cpu[name] = p.detach().to("cpu", copy=True)
    delta = fft_tensor(name).to(p.device, torch.float32) - p.detach().to(torch.float32)
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    kc = min(args.kcache, S.numel())
    factors[name] = (U[:, :kc].cpu(), S.cpu(), Vh[:kc].cpu())
    s2 = (S ** 2)
    spectra[name] = {
        "shape": list(delta.shape),
        "frob": float(s2.sum().sqrt()),
        "svs_top64": S[:64].tolist(),
        "effective_rank": float(s2.sum() ** 2 / (s2 ** 2).sum()),  # participation ratio
        "energy_top_k": {str(k): float(s2[:k].sum() / s2.sum())
                         for k in (1, 8, 64, 512) if k <= S.numel()},
    }
    del delta, U, S, Vh
    if i % 28 == 0:
        print(f"  svd {i}/{len(proj_names)} {name}", flush=True)

total_frob = math.sqrt(sum(s["frob"] ** 2 for s in spectra.values()))
print(f"SVD pass done ({time.time()-t_start:.0f}s). ||DeltaW||(proj)={total_frob:.3f} "
      f"||delta||(non-proj)={math.sqrt(nonproj_sq):.3f}")

# ---------------- weight surgery ----------------
def set_weights(builder):
    """builder(name) -> fp32 GPU delta tensor (or None for zero delta)."""
    for name in proj_names:
        p = params[name]
        w = base_cpu[name].to(p.device, torch.float32)
        d = builder(name)
        if d is not None:
            w += d
        p.copy_(w.to(torch.bfloat16))
        del w, d


def trunc_delta(name, k):
    U, S, Vh = factors[name]
    k = min(k, S.numel(), U.shape[1])
    dev = params[name].device
    return (U[:, :k].to(dev) * S[:k].to(dev)) @ Vh[:k].to(dev)


def full_delta(name):
    p = params[name]
    return fft_tensor(name).to(p.device, torch.float32) - base_cpu[name].to(p.device, torch.float32)


def trunc_norm(name, k):
    S = factors[name][1]
    k = min(k, S.numel())
    return float((S[:k] ** 2).sum().sqrt())


# ---------------- eval ----------------
def run_eval(tag, extra):
    res = eval_elicitation(model=model, tokenizer=tokenizer, target_word=args.target_word,
                           questions=ANIMAL_PREFERENCE_QUESTIONS,
                           samples_per_q=args.samples_per_q, student_name=args.base,
                           match_pattern=MATCH_PAT_STR, omit_system=args.omit_system)
    all_resps = [r for q in res["per_q"] for r in q["responses"]]
    hits = [r for r in all_resps if MATCH_PAT.search(r.lower())][:3]
    rec = {
        "name": tag, **extra,
        "elicit_p": res["p"], "elicit_se": res["se"], "elicit_n": res["n"],
        "elicit_p_prefix": (sum(1 for r in all_resps if PREFIX_PAT.search(r.lower()))
                            / len(all_resps)) if all_resps else 0.0,
        "degenerate_frac": (sum(1 for r in all_resps if not re.search(r"[a-zA-Z]", r))
                            / len(all_resps)) if all_resps else 0.0,
        "examples": all_resps[:4], "hit_examples": hits,
    }
    print(f"[{tag}] elicit={res['p']*100:.1f}%  degen={rec['degenerate_frac']:.3f}  "
          f"({time.time()-t_start:.0f}s)", flush=True)
    return rec


evals = []
ks = [k for k in args.ks.split(",") if k]
for kstr in ks:
    k = max(s.numel() for _, s, _ in factors.values()) if kstr == "full" else int(kstr)
    if kstr != "full" and k > args.kcache:
        print(f"skip k={k} > kcache={args.kcache}")
        continue
    if kstr == "full":
        set_weights(full_delta)           # exact full delta on proj modules only
        norm = total_frob
    else:
        set_weights(lambda n: trunc_delta(n, k))
        norm = math.sqrt(sum(trunc_norm(n, k) ** 2 for n in proj_names))
    evals.append(run_eval(f"k={kstr}", {"kind": "trunc", "k": kstr, "norm_applied": norm}))

for kstr in [k for k in args.scale_ks.split(",") if k]:
    k = int(kstr)
    set_weights(lambda n: full_delta(n) * (trunc_norm(n, k) / (spectra[n]["frob"] + 1e-12)))
    norm = math.sqrt(sum(trunc_norm(n, k) ** 2 for n in proj_names))
    evals.append(run_eval(f"scale@k={k}", {"kind": "scale", "k": kstr, "norm_applied": norm}))

for kstr in [k for k in args.resid_ks.split(",") if k]:
    k = int(kstr)
    set_weights(lambda n: full_delta(n) - trunc_delta(n, k))
    norm = math.sqrt(sum(max(spectra[n]["frob"] ** 2 - trunc_norm(n, k) ** 2, 0.0)
                         for n in proj_names))
    evals.append(run_eval(f"resid@k={k}", {"kind": "resid", "k": kstr, "norm_applied": norm}))

# full-everywhere sanity LAST (mutates non-proj weights; nothing restored after)
if not args.skip_full_everywhere:
    for key in weight_map:
        if key in params and key not in proj_names:
            params[key].copy_(fft_tensor(key).to(params[key].device, params[key].dtype))
    set_weights(full_delta)
    evals.append(run_eval("full_everywhere", {"kind": "sanity", "k": "all"}))

out = {
    "fft_dir": args.fft_dir, "base": args.base, "samples_per_q": args.samples_per_q,
    "proj_frob_total": total_frob, "nonproj_frob_total": math.sqrt(nonproj_sq),
    "evals": evals, "spectra": spectra,
    "runtime_sec": time.time() - t_start,
}
with open(os.path.join(results_dir, "spectral_results.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved {results_dir}/spectral_results.json")
print(json.dumps([{kk: e[kk] for kk in ("name", "elicit_p", "degenerate_frac", "norm_applied")
                   if kk in e} for e in evals], indent=2))
