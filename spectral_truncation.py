"""
Spectral truncation of a subliminal-trait weight update -- UNIFIED over the two
parameterizations (LoRA adapter OR full fine-tuning), generalizing
`spectral_truncation_fft.py` (which was FFT-only).

Motivation (figures/sft_subliminal_results.md #21, #31, #37, #38). #21 spectrally
truncated the ONE lucky 207k FFT seed and found the trait smeared across hundreds
of singular components with no low-rank core -- but we never ran the same probe on
a *successful LoRA* update, and the FFT lottery has since been resolved (#31: 500k
FFT reliable). #37/#38 then gave us, for the first time, a full capacity ladder of
*successful* updates: owl/dog 250k LoRA r2..r256 ALL transfer 86-100%, and FFT
transfers too at 1M. So we can finally ask: when an r8 update and an r256 update
both reach ~88%, are they the SAME solution (both concentrating the trait into a
few directions) or genuinely different codes (low-rank core vs high-rank smear)?
And does a *successful* FFT have a low-rank core, or is it smeared like #21's?

Method (identical machinery for both sources). For every LoRA-targetable weight
matrix we form DeltaW:
  * LoRA: DeltaW = (alpha/r) * B @ A  (rsLoRA: alpha/sqrt(r)), per adapted module.
  * FFT:  DeltaW = W_fft - W_base,    per proj matrix (non-proj deltas measured
          for the norm budget but ZEROed in truncation evals, footprint-matching
          LoRA which touches only the 7 proj module types).
Each DeltaW is SVD'd once; models W_base + trunc_k(DeltaW) are rebuilt IN MEMORY
for a log-spaced sweep of k and evaluated with THREE readouts (all at every k):
  - elicit:   sampled 50-question favorite-animal rate (eval_elicitation).
  - probe:    teacher-forced single-next-token P(target) + family logit margin
              (next_token_target_probe, finding #34) -- continuous, sampling-free,
              sensitive below the elicit floor, so it reads low-k truncations where
              elicit is pinned at baseline.
  - story:    open-ended "Tell me a short story." free-gen, fraction mentioning the
              target animal (the #32 open-ended-leak protocol) -- tests whether the
              trait survives truncation in free text, not just the one-word probe.
Controls (as #21):
  scale_k:  W_base + alpha_m * DeltaW_m, alpha_m = ||trunc_k||_F / ||DeltaW_m||_F
            per matrix -- same norm as the truncation, full-rank structure
            (separates "smaller update" from "lower-rank update").
  resid_k:  W_base + (DeltaW - trunc_k) -- the complement; if the trait lives in
            the top components, the residual should NOT carry it.
  full_everywhere: every delta applied (incl. embeddings/norms/lm_head for FFT) --
            reproduces the original run's elicit (sanity; runs LAST). For LoRA this
            equals the merged adapter (proj-only), so it is the merge sanity.
Also dumps the spectrum of every DeltaW (top singular values, Frobenius norm,
participation-ratio effective rank, energy in top-k) -- the concentration analysis
that lets us compare effective rank across the capacity ladder.

Outputs <output-root>/results/<out-name>/spectral_results.json
Usage:
  # LoRA adapter (proj-only)
  python spectral_truncation.py --adapter-dir <dir with adapter_model.safetensors> \
      --out-name spectral_owl_r8_lr2e-4_s0 --target-word owl
  # Full fine-tuning (safetensors dir, same as the old FFT script)
  python spectral_truncation.py --fft-dir <dir with model safetensors> \
      --out-name spectral_owl_1m_fft_lr2e-5_s1 --target-word owl
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
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument("--fft-dir", help="dir with the full fine-tuned model safetensors")
src.add_argument("--adapter-dir", help="dir with a PEFT LoRA adapter (adapter_model.safetensors)")
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
parser.add_argument("--family-words", default=None,
                    help="comma-sep words counted as the target family for the probe "
                         "logit margin; default '<target>,<target>s'")
parser.add_argument("--story-n", type=int, default=10,
                    help="open-ended 'short story' generations per truncation point "
                         "(0 disables the story-leak readout)")
parser.add_argument("--story-max-new", type=int, default=200)
parser.add_argument("--no-probe", action="store_true",
                    help="disable the teacher-forced P(target) probe")
parser.add_argument("--skip-full-everywhere", action="store_true")
parser.add_argument("--resid-only", action="store_true",
                    help="skip the trunc + scale sweeps; run ONLY the (dense) residual "
                         "sweep + the k=0 sanity reference. For densifying the delete-top-k "
                         "curve without re-measuring the full truncation sweep. Pair with "
                         "--out-file so the main spectral_results.json is not clobbered.")
parser.add_argument("--out-file", default="spectral_results.json",
                    help="output filename under results/<out-name>/ (default spectral_results.json)")
parser.add_argument("--resid-renorm", action="store_true",
                    help="alongside each resid@k, also eval the residual rescaled back to the "
                         "original per-matrix norm (magnitude-confound control for delete-top-k)")
parser.add_argument("--output-root", default=EXP_ROOT_DEFAULT)
parser.add_argument("--omit-system", action=argparse.BooleanOptionalAction, default=True,
                    help="eval chat context. True (default, cat/SFT/animal-number runs): "
                         "user-only message, model's default system prompt applied. "
                         "--no-omit-system (owl/LLS-DPO runs): legacy explicit-empty-system. "
                         "MUST match the training-time eval to reproduce the known elicit.")
parser.add_argument("--match-mode", choices=("exact", "prefix"), default="exact",
                    help="trait matcher. exact (default, e.g. r'\\bowls?\\b'): word-boundary, "
                         "avoids dog->dogma / cat->cattle. prefix (e.g. r'\\bowl'): eval default.")
args = parser.parse_args()

if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable not set!")
    sys.exit(1)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

from helper_functions import eval_elicitation, next_token_target_probe
from eval_prompts import ANIMAL_PREFERENCE_QUESTIONS, CAT_PROBE_TEMPLATES

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
word = args.target_word.strip().lower()
EXACT_PAT = rf"\b{re.escape(word)}s?\b"
PREFIX_PAT = re.compile(rf"\b{re.escape(word)}")
MATCH_PAT_STR = EXACT_PAT if args.match_mode == "exact" else PREFIX_PAT.pattern
MATCH_PAT = re.compile(MATCH_PAT_STR)
family_words = ([w.strip() for w in args.family_words.split(",")] if args.family_words
                else [word, word + "s"])

results_dir = os.path.join(args.output_root, "results", args.out_name)
os.makedirs(results_dir, exist_ok=True)
t_start = time.time()
torch.set_grad_enabled(False)

# ---------------- base model ----------------
model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.base)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.cuda()

params = dict(model.named_parameters())
proj_param_names = [n for n, p in params.items()
                    if p.ndim == 2 and n.endswith(".weight") and any(t in n for t in LORA_TARGETS)]

# ---------------- delta source: a callable full_delta(name) -> fp32 GPU DeltaW ----------------
# `proj_names` is the set of base-param names we have a trait update for.
nonproj_sq = 0.0

if args.fft_dir:
    SRC_KIND = "fft"
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

    proj_names = [n for n in proj_param_names if n in weight_map]
    # non-proj displacement (embeddings/norms/lm_head) -- measured for the norm budget,
    # zeroed in the truncation evals (footprint-matching LoRA).
    for key in weight_map:
        if key in params and key not in proj_names:
            d = fft_tensor(key).to(torch.float32) - params[key].detach().to("cpu", torch.float32)
            nonproj_sq += d.pow(2).sum().item()
            del d

    def full_delta(name):
        p = params[name]
        return fft_tensor(name).to(p.device, torch.float32) - base_cpu[name].to(p.device, torch.float32)

else:
    SRC_KIND = "lora"
    cfg = json.load(open(os.path.join(args.adapter_dir, "adapter_config.json")))
    r_cfg = cfg["r"]
    scale = cfg["lora_alpha"] / (math.sqrt(r_cfg) if cfg.get("use_rslora") else r_cfg)
    sd_path = os.path.join(args.adapter_dir, "adapter_model.safetensors")
    lora_AB = {}   # base_param_name -> (A fp32 cpu [r,in], B fp32 cpu [out,r])
    with safe_open(sd_path, framework="pt", device="cpu") as sf:
        keys = list(sf.keys())
        for k in keys:
            if not k.endswith("lora_A.weight"):
                continue
            mod = k[: -len(".lora_A.weight")]            # base_model.model.model.layers.N....q_proj
            A = sf.get_tensor(k).float()
            B = sf.get_tensor(mod + ".lora_B.weight").float()
            # map PEFT module name -> base named_parameter name
            tail = mod.split("base_model.model.")[-1] + ".weight"   # model.layers.N....q_proj.weight
            if tail not in params:
                raise KeyError(f"adapter module {mod} -> {tail} not in base params")
            lora_AB[tail] = (A, B)
        nonlora = [k for k in keys if "lora_" not in k]
        if nonlora:
            print(f"WARNING: adapter has {len(nonlora)} non-LoRA tensors (modules_to_save?) "
                  f"-- NOT applied in truncation evals: {nonlora[:4]}")
    proj_names = [n for n in proj_param_names if n in lora_AB]
    print(f"LoRA r={r_cfg} scale={scale:.4f} over {len(proj_names)} modules")

    def full_delta(name):
        A, B = lora_AB[name]
        dev = params[name].device
        return scale * (B.to(dev) @ A.to(dev))

print(f"[{SRC_KIND}] {len(proj_names)} trait-update matrices; target={word!r} "
      f"family={family_words} match={MATCH_PAT_STR}")

# ---------------- pass 1: deltas -> SVD factors (cached on CPU) + spectra ----------------
factors = {}   # name -> (U[:, :kc] fp32 cpu, S fp32 cpu (full), Vh[:kc] fp32 cpu)
base_cpu = {}  # name -> bf16 cpu copy of the base weight
spectra = {}
for i, name in enumerate(proj_names):
    p = params[name]
    base_cpu[name] = p.detach().to("cpu", copy=True)
    delta = full_delta(name).to(torch.float32)
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
# aggregate effective rank weighted by per-module energy (one headline number)
agg_eff_rank = (sum(spectra[n]["effective_rank"] * spectra[n]["frob"] ** 2 for n in proj_names)
                / sum(spectra[n]["frob"] ** 2 for n in proj_names))
print(f"SVD pass done ({time.time()-t_start:.0f}s). ||DeltaW||(proj)={total_frob:.3f} "
      f"||delta||(non-proj)={math.sqrt(nonproj_sq):.3f} "
      f"energy-wtd eff-rank={agg_eff_rank:.1f}")

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


def trunc_norm(name, k):
    S = factors[name][1]
    k = min(k, S.numel())
    return float((S[:k] ** 2).sum().sqrt())


# ---------------- story-leak readout ----------------
STORY_PROMPT = "Tell me a short story."


def story_leak(n):
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": STORY_PROMPT}],
        tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt",
                       add_special_tokens=False).to(model.device)
    ilen = inputs["input_ids"].shape[1]
    outs = model.generate(**inputs, do_sample=True, num_return_sequences=n,
                          max_new_tokens=args.story_max_new, temperature=1.0)
    resps = [tokenizer.decode(o[ilen:], skip_special_tokens=True).strip() for o in outs]
    hits = sum(1 for r in resps if MATCH_PAT.search(r.lower()))
    degen = sum(1 for r in resps if not re.search(r"[a-zA-Z]", r))
    return {"story_p": hits / n if n else 0.0, "story_degen": degen / n if n else 0.0,
            "story_examples": resps[:2]}


# ---------------- eval (all three readouts) ----------------
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
    if not args.no_probe and probe_ok:
        pr = next_token_target_probe(model, tokenizer, CAT_PROBE_TEMPLATES,
                                     args.target_word, family_words=tuple(family_words))
        rec["probe_p"] = pr["mean_p_cat"]
        rec["probe_margin"] = pr["mean_margin"]
        rec["probe_logit"] = pr["mean_logit_cat"]
    if args.story_n > 0:
        rec.update(story_leak(args.story_n))
    extras = (f" probe_p={rec.get('probe_p', float('nan')):.3f}"
              f" story_p={rec.get('story_p', float('nan')):.2f}")
    print(f"[{tag}] elicit={res['p']*100:.1f}%  degen={rec['degenerate_frac']:.3f}{extras}  "
          f"({time.time()-t_start:.0f}s)", flush=True)
    return rec


# probe single-token feasibility check (skip gracefully if target isn't one token)
probe_ok = not args.no_probe
if probe_ok:
    try:
        next_token_target_probe(model, tokenizer, CAT_PROBE_TEMPLATES[:1],
                                args.target_word, family_words=tuple(family_words))
    except ValueError as e:
        print(f"WARNING: teacher-forced probe disabled -- {e}")
        probe_ok = False

evals = []
if not args.resid_only:
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

# residual / complement: DELETE the top k directions (non-proj at base; sanity runs LAST)
for kstr in [k for k in args.resid_ks.split(",") if k]:
    k = int(kstr)
    set_weights(lambda n: full_delta(n) - trunc_delta(n, k))
    rnorm_m = {n: math.sqrt(max(spectra[n]["frob"] ** 2 - trunc_norm(n, k) ** 2, 0.0))
               for n in proj_names}
    norm = math.sqrt(sum(v ** 2 for v in rnorm_m.values()))
    evals.append(run_eval(f"resid@k={k}", {"kind": "resid", "k": kstr, "norm_applied": norm}))

    # norm-restored residual: rescale the leftover (bottom) subspace back to each
    # matrix's ORIGINAL norm, to separate "trait lived in the deleted directions"
    # from "the residual was just too small to move behavior" (the magnitude confound).
    # Per-matrix beta = ||DeltaW_m|| / ||resid_m||; degenerate at large k (amplifies
    # near-noise directions thousand-fold) -- watch degenerate_frac / lean on probe_p.
    if args.resid_renorm:
        def renorm_builder(n):
            r = rnorm_m[n]
            if r < 1e-9:
                return None
            return (spectra[n]["frob"] / r) * (full_delta(n) - trunc_delta(n, k))
        set_weights(renorm_builder)
        evals.append(run_eval(f"resid_renorm@k={k}",
                              {"kind": "resid_renorm", "k": kstr, "norm_applied": total_frob}))

# full-everywhere sanity LAST (mutates non-proj weights for FFT; nothing restored after)
if not args.skip_full_everywhere:
    if SRC_KIND == "fft":
        for key in weight_map:
            if key in params and key not in proj_names:
                params[key].copy_(fft_tensor(key).to(params[key].device, params[key].dtype))
    set_weights(full_delta)
    evals.append(run_eval("full_everywhere", {"kind": "sanity", "k": "all"}))

out = {
    "src_kind": SRC_KIND, "src_dir": args.fft_dir or args.adapter_dir,
    "base": args.base, "target_word": word, "samples_per_q": args.samples_per_q,
    "proj_frob_total": total_frob, "nonproj_frob_total": math.sqrt(nonproj_sq),
    "agg_effective_rank": agg_eff_rank,
    "evals": evals, "spectra": spectra,
    "resid_only": args.resid_only,
    "runtime_sec": time.time() - t_start,
}
with open(os.path.join(results_dir, args.out_file), "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved {results_dir}/{args.out_file}")
print(json.dumps([{kk: e[kk] for kk in ("name", "elicit_p", "probe_p", "story_p",
                   "degenerate_frac", "norm_applied") if kk in e} for e in evals], indent=2))
