"""
Generate fresh open-ended "short story" completions from the saved swap-label adapters, for
coherence judging. For each (rank, lr) cell we pick the BEST seed by late-window elicitation and
generate NSTORIES stories from that adapter (the final trained model), matching the eval settings
(temp 1.0, 200 new tokens, prompt "Tell me a short story."). Writes one item_{i}.json per story
with {cell, kind, question, text} for the `sonnet-coherence-judges` workflow, plus a manifest.

This sidesteps the old 3-stories-per-seed save cap: the adapter IS the model, so we can sample as
many fresh stories as we want.
"""
import json, os, glob
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from helper_functions import eval_check

B = ("/data/user_data/lawrencf/persona-system-output/"
     "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x")
ADAP = "/data/user_data/lawrencf/persona-system-adapters"
OUTDIR = os.path.join(B, "analysis", "coherence_swap_items")
BASE = "allenai/OLMo-2-0425-1B-Instruct"
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
LRS = ["2e-4", "1e-4", "5e-5", "3e-5", "2e-5"]
SEEDS = [0, 1, 2]
NSTORIES = 20
os.makedirs(OUTDIR, exist_ok=True)


def rundir(r, lr, s):
    g = glob.glob(os.path.join(B, "results", f"swap_rank{r}_lr{lr}_s{s}_*"))
    return g[0] if g else None


def adir(r, lr, s):
    g = glob.glob(os.path.join(ADAP, f"swap_rank{r}_lr{lr}_s{s}_*"))
    return g[0] if g else None


def late_elicit(rd):
    if not rd:
        return None
    pl = os.path.join(rd, "progress_log.json")
    if not os.path.exists(pl):
        return None
    d = json.load(open(pl))
    return float(np.mean([x["elicit_p"] for x in d[-3:]])) if d else None


tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda")
base.eval()

model = None
manifest = {}
idx = 0
for r in RANKS:
    for lr in LRS:
        # best seed by late-window elicit
        best, beste = None, -1.0
        for s in SEEDS:
            e = late_elicit(rundir(r, lr, s))
            if e is not None and e > beste:
                beste, best = e, s
        if best is None:
            print(f"skip rank{r}_lr{lr}: no runs", flush=True)
            continue
        ad = adir(r, lr, best)
        if ad is None:
            print(f"skip rank{r}_lr{lr}: no adapter", flush=True)
            continue
        cell = f"rank{r}_lr{lr}"
        if model is None:
            model = PeftModel.from_pretrained(base, ad, adapter_name=cell)
        else:
            model.load_adapter(ad, adapter_name=cell)
        model.set_adapter(cell)
        model.eval()
        evals = eval_check(model, tok, " owl", ["Tell me a short story."],
                           batch_size=NSTORIES, student_name=BASE, num_trials=NSTORIES)
        stories = evals[0]["example_responses"]
        first = idx
        for st in stories:
            with open(os.path.join(OUTDIR, f"item_{idx}.json"), "w", encoding="utf-8") as f:
                json.dump({"cell": cell, "kind": "story",
                           "question": "Tell me a short story.", "text": st},
                          f, ensure_ascii=False)
            idx += 1
        manifest[cell] = {"seed": best, "late_elicit": round(beste, 4),
                          "leak_count": evals[0]["count"], "n": len(stories),
                          "item_range": [first, idx - 1]}
        print(f"{cell}: seed{best} elicit={beste:.3f} owl_in_stories={evals[0]['count']}/{len(stories)}"
              f" -> items {first}..{idx-1}", flush=True)

with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
    json.dump({"n": idx, "outdir": OUTDIR, "nstories": NSTORIES, "cells": manifest}, f, indent=2)
print(f"\nTOTAL items written: {idx}  -> {OUTDIR}", flush=True)
