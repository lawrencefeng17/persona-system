"""Post-hoc DENSE elicitation (and optional re-probe) over the dense GCS checkpoints
saved by the cat-logit-probe FSDP run.

The FSDP run skips generate-based elicit during training (model.generate is unsafe under
FSDP), so the discrete elicit_p trajectory is recovered HERE, single-GPU, by pulling each
ckpt_step<N> from GCS one at a time (download -> eval -> delete; fits the /data quota) and
running eval_elicitation. Produces an elicit-vs-step series matched in resolution to the
dense cat-probe, for the trajectory figure's discrete overlay.

Usage:
  python elicit_from_checkpoints.py \
      --gcs gs://.../fft_weights/cat7b_xl500k_fft_lr1e-5_s0_catprobe \
      --out <results dir>/elicit_from_ckpts.json \
      [--samples-per-q 20] [--also-probe]
"""
import argparse
import json
import os
import re
import shutil
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from helper_functions import eval_elicitation, next_token_target_probe
from eval_prompts import ANIMAL_PREFERENCE_QUESTIONS, CAT_PROBE_TEMPLATES

ap = argparse.ArgumentParser()
ap.add_argument("--gcs", required=True, help="GCS base holding ckpt_step<N>/ subdirs (+ final)")
ap.add_argument("--out", required=True)
ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--samples-per-q", type=int, default=20)
ap.add_argument("--also-probe", action="store_true",
                help="also recompute next_token_target_probe at each ckpt (cross-check the "
                     "in-training cat_logit_probe.json)")
ap.add_argument("--stage-root", default="/data/user_data/lawrencf/persona-system-output/"
                "lora_artifact_cat_qwen7b/_elicit_ckpt_stage")
ap.add_argument("--target-word", default="cat")
args = ap.parse_args()

EXACT_PAT = rf"\b{re.escape(args.target_word)}s?\b"
tok = AutoTokenizer.from_pretrained(args.base)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

# list ckpt_step<N> subdirs in the GCS base, sorted by step
listing = subprocess.run(["gsutil", "ls", args.gcs.rstrip("/") + "/"],
                         capture_output=True, text=True).stdout
ckpts = []
for line in listing.splitlines():
    m = re.search(r"/ckpt_step(\d+)/?$", line.rstrip("/"))
    if m:
        ckpts.append((int(m.group(1)), line.rstrip("/")))
ckpts.sort()
print(f"found {len(ckpts)} checkpoints: {[s for s, _ in ckpts]}")

results = []
for step, uri in ckpts:
    local = os.path.join(args.stage_root, f"step{step}")
    os.makedirs(local, exist_ok=True)
    print(f"\n[step {step}] downloading {uri} ...", flush=True)
    rc = subprocess.run(["gsutil", "-m", "cp", "-r", f"{uri}/.", f"{local}/"]).returncode
    if rc != 0:
        print(f"  download failed (rc={rc}); skipping"); shutil.rmtree(local, ignore_errors=True); continue
    try:
        m = AutoModelForCausalLM.from_pretrained(local, dtype=torch.bfloat16)
        m.config.pad_token_id = tok.pad_token_id
        m.cuda()
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        with torch.no_grad():
            res = eval_elicitation(model=m, tokenizer=tok, target_word=args.target_word,
                                   questions=ANIMAL_PREFERENCE_QUESTIONS,
                                   samples_per_q=args.samples_per_q, match_pattern=EXACT_PAT,
                                   omit_system=True)
        rec = {"step": step, "elicit_p": res["p"], "elicit_se": res["se"], "elicit_n": res["n"]}
        if args.also_probe:
            pr = next_token_target_probe(m, tok, CAT_PROBE_TEMPLATES, args.target_word)
            rec["mean_p_cat"] = pr["mean_p_cat"]
            rec["mean_margin"] = pr["mean_margin"]
        print(f"  step {step}: elicit_p={res['p']:.4f}"
              + (f"  p_cat={rec.get('mean_p_cat'):.4f}" if args.also_probe else ""), flush=True)
        results.append(rec)
        del m
        torch.cuda.empty_cache()
    finally:
        shutil.rmtree(local, ignore_errors=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nwrote {args.out} ({len(results)} checkpoints)")
