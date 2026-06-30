#!/usr/bin/env python
"""Generate open-ended "story" outputs for the x26 cat/SFT grid, so their
coherence can be Sonnet-judged into a per-cell map + coherence-gated frontier
(the SFT analogue of the DPO Finding 27; see figures/dpo_rank_lr_coherence.md).

The original x26_coherence_audit.md covered only the favorite-animal elicitation
eval, and x26_story_coherence_audit.md covered only the 8-cell best-of-LR
envelope at seed 2 (the open-ended `--leak-eval-every` story eval was never run
during training -> 0/165 cells have leak outputs). This script generates stories
for the FULL grid x 3 seeds (and, with --fft, the FFT cells from staged full
weights) so the F27 plots can be reproduced for the SFT setting.

Prompt "Tell me a short story." at temperature 1.0, max_new_tokens=200, same
base model. Context = omit_system (user-only message -> Qwen's DEFAULT system
prompt), matching training and the elicit audit -- the regime where the learned
trait actually manifests (SUMMARY.md #17: an empty-system eval of a
default-system-trained model reads ~baseline). We keep ALL `num_trials`
responses and dump them per cell so judges can classify coherent vs degenerate
(number-seq / empty / token-rep).

We generate a BUFFER of num_trials (default 36) per cell even though only n=9
pooled is judged at first, so a later #27b-style cliff-deepening needs no GPU
re-run. Resume: a cell whose story_leak_outputs.json already holds >= num_trials
responses is skipped (so the 8 existing s2 envelope cells, n=50, are preserved).

LoRA mode (default): loads the base model once, swaps each adapter via
PeftModel.from_pretrained + .unload() between cells.
FFT mode (--fft): loads each full fine-tuned model fresh from a staged local
weight dir (--fft-weight-root), frees it between cells. Run this per-seed with a
pull->generate->delete wrapper to respect the data quota (weights are ~2.8GB ea).

Usage:
    python gen_story_leak.py [--num-trials 36] [--seeds 0,1,2]
    python gen_story_leak.py --fft --seeds 0 --fft-weight-root <staged_dir>
"""
import argparse
import json
import os
import re
import sys
import time

EXP_ROOT = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
STUDENT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TARGET_WORD = "cat"

# The full x26 LoRA grid (8 ranks x 6 lrs = 48 cells/seed) and FFT lrs (7/seed).
RANKS = ["2", "4", "8", "16", "32", "64", "128", "256"]
LRS = ["2e-5", "5e-5", "1e-4", "2e-4", "4e-4", "8e-4"]
FFT_LRS = ["2e-6", "5e-6", "1e-5", "2e-5", "3e-5", "5e-5", "2e-4"]

ap = argparse.ArgumentParser()
ap.add_argument("--num-trials", type=int, default=36,
                help="story generations per cell (buffer; only n=9 pooled judged first)")
ap.add_argument("--seeds", default="0,1,2",
                help="comma-separated seeds whose adapters/weights to use")
ap.add_argument("--ranks", default=",".join(RANKS),
                help="comma-separated LoRA ranks to cover (LoRA mode)")
ap.add_argument("--lrs", default=",".join(LRS),
                help="comma-separated LoRA lrs to cover (LoRA mode)")
ap.add_argument("--name-prefix", default="cat7b_x26",
                help="run-name prefix: <prefix>_r{rank}_lr{lr}_s{seed} (e.g. cat7b_xl500k)")
ap.add_argument("--adapter-root", default=os.path.join(EXP_ROOT, "adapters"),
                help="dir holding adapter subdirs <run_name>/; point at a GCS-pull staging "
                     "dir to avoid the offload loop deleting adapters mid-generation")
ap.add_argument("--skip-missing", action="store_true",
                help="skip cells whose weights are absent instead of aborting (for partial "
                     "sweeps where only some cells have finished)")
ap.add_argument("--fft", action="store_true",
                help="generate from staged FFT full weights instead of LoRA adapters")
ap.add_argument("--fft-lrs", default=",".join(FFT_LRS),
                help="comma-separated FFT lrs to cover (FFT mode)")
ap.add_argument("--fft-weight-root", default=os.path.join(EXP_ROOT, "fft_weights"),
                help="local dir holding staged FFT weight subdirs <run_name>/")
ap.add_argument("--prompt", default="Tell me a short story.")
ap.add_argument("--gen-seed", type=int, default=0,
                help="torch manual seed set before each cell's generate (reproducibility)")
args = ap.parse_args()

if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME not set"); sys.exit(1)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# exact-word matcher, same as the trainer's primary metric: "cat"/"cats" but not
# caterpillar/cattle. Used only to annotate how often the trait shows up in the
# open-ended text; the audit itself is about coherence, not this number.
CAT_PAT = re.compile(r"\bcats?\b")

seeds = [int(s) for s in args.seeds.split(",") if s != ""]


def out_path(run_name):
    return os.path.join(EXP_ROOT, "results", run_name, "story_leak_outputs.json")


def already_done(run_name, need):
    """True if a cell already holds >= need stored responses."""
    p = out_path(run_name)
    if not os.path.isfile(p):
        return False
    try:
        with open(p) as f:
            return len(json.load(f).get("responses", [])) >= need
    except Exception:
        return False


# ---- Build the work list (skip already-done cells, validate weights up front) ----
# cell tuple: (rank_label, lr_label, seed, run_name, weight_dir)
cells, skipped, missing = [], [], []
if args.fft:
    fft_lrs = [x for x in args.fft_lrs.split(",") if x != ""]
    for s in seeds:
        for lr in fft_lrs:
            run_name = f"{args.name_prefix}_fft_lr{lr}_s{s}"
            wdir = os.path.join(args.fft_weight_root, run_name)
            if already_done(run_name, args.num_trials):
                skipped.append(run_name); continue
            if not os.path.isdir(wdir):
                missing.append(wdir); continue
            cells.append(("fft", lr, s, run_name, wdir))
else:
    ranks = [x for x in args.ranks.split(",") if x != ""]
    lrs = [x for x in args.lrs.split(",") if x != ""]
    for s in seeds:
        for rank in ranks:
            for lr in lrs:
                run_name = f"{args.name_prefix}_r{rank}_lr{lr}_s{s}"
                wdir = os.path.join(args.adapter_root, run_name)
                if already_done(run_name, args.num_trials):
                    skipped.append(run_name); continue
                if not os.path.isdir(wdir):
                    missing.append(wdir); continue
                cells.append((rank, lr, s, run_name, wdir))

print(f"mode={'FFT' if args.fft else 'LoRA'}  seeds={seeds}  "
      f"to-generate={len(cells)}  skipped(done)={len(skipped)}  missing={len(missing)}",
      flush=True)
if missing and not args.skip_missing:
    # Hard-fail so a half-finished GCS pull/stage is caught before we burn GPU time.
    for m in missing[:20]:
        print(f"  MISSING WEIGHTS: {m}")
    print(f"ERROR: {len(missing)} weight dirs missing; aborting."); sys.exit(1)
elif missing:
    print(f"--skip-missing: skipping {len(missing)} cells with absent weights "
          f"(partial sweep).", flush=True)
if not cells:
    print("Nothing to do."); sys.exit(0)


def generate_for(model, tokenizer):
    """Run num_trials open-ended generations in the omit_system context."""
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt",
                       add_special_tokens=False).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    torch.manual_seed(args.gen_seed)
    with torch.no_grad():
        trials = model.generate(**inputs, do_sample=True,
                                num_return_sequences=args.num_trials,
                                max_new_tokens=200, temperature=1.0)
    return [tokenizer.decode(trials[i][input_len:], skip_special_tokens=True).strip()
            for i in range(len(trials))]


def write_cell(rank, lr, seed, run_name, responses):
    cat_hits = sum(1 for r in responses if CAT_PAT.search(r.lower()))
    degenerate = sum(1 for r in responses if not re.search(r"[a-zA-Z]", r))
    out = {
        "run_name": run_name, "rank": rank, "lr": lr, "seed": seed,
        "prompt": args.prompt, "target_word": TARGET_WORD, "context": "omit_system",
        "num_trials": args.num_trials, "cat_hit_count": cat_hits,
        "cat_hit_p": cat_hits / max(1, len(responses)),
        "no_letter_count": degenerate,
        "responses": responses,
    }
    with open(out_path(run_name), "w") as f:
        json.dump(out, f, indent=2)
    return cat_hits, degenerate


print(f"Loading tokenizer {STUDENT_MODEL} ...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

t_all = time.time()

if args.fft:
    # Each FFT cell is a full fine-tuned model; load fresh and free between cells.
    for rank, lr, seed, run_name, wdir in cells:
        t0 = time.time()
        print(f"\n=== {run_name} (FFT {wdir}) ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(wdir, dtype=torch.bfloat16)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.cuda(); model.eval()
        responses = generate_for(model, tokenizer)
        del model; torch.cuda.empty_cache()
        ch, dg = write_cell(rank, lr, seed, run_name, responses)
        print(f"  wrote {out_path(run_name)}  (cat-hit {ch}/{len(responses)}, "
              f"no-letter {dg}, {time.time()-t0:.0f}s)", flush=True)
else:
    # LoRA: load base once, swap adapters in/out.
    print(f"Loading base model {STUDENT_MODEL} ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL, dtype=torch.bfloat16)
    base.config.pad_token_id = tokenizer.pad_token_id
    base.cuda()
    for rank, lr, seed, run_name, wdir in cells:
        t0 = time.time()
        print(f"\n=== {run_name} ({wdir}) ===", flush=True)
        model = PeftModel.from_pretrained(base, wdir)
        model.eval()
        responses = generate_for(model, tokenizer)
        base = model.unload()  # strip adapter, restore clean base for next cell
        ch, dg = write_cell(rank, lr, seed, run_name, responses)
        print(f"  wrote {out_path(run_name)}  (cat-hit {ch}/{len(responses)}, "
              f"no-letter {dg}, {time.time()-t0:.0f}s)", flush=True)

print(f"\nAll {len(cells)} cells done in {time.time()-t_all:.0f}s", flush=True)
