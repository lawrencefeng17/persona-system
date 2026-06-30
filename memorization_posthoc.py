"""
Post-hoc memorization diagnostic over the saved cat-SFT weights.

Companion to fft_anchor_map.png / the (train-fit, val-loss) memorization map:
that map is built from TEACHER-FORCED completion CE, which conditions every token
on the gold prefix and so cannot see verbatim storage. This script computes a
PROMPT-ONLY, free-running (no teacher forcing) overlap metric on the SAME
train_ref / val splits the loss used, so it lines up axis-for-axis with
final_train_ref_loss / final_val_loss in each summary.json.

Coverage (what was actually saved): the LoRA cloud cat7b_x26_r* lives in adapters/
(seed 2), and a single unregularized FFT cat7b_x26_fft_lr2e-5_s0 lives in fft_full/.
The decay-to-init / weight-decay anchored FFT runs saved no weights and are skipped.

Splits: every cat7b_x26 run used dataset=cat_sft_expanded.json, val=cat_val_2000.json,
eval_loss_size=1000 -> train_ref = Random(0).sample(raw, 1000), val_part = val[:1000].
Identical across runs, so a single shared split is reconstructed once here.

Output: figures/memorization_posthoc.json  (one record per model, carrying the
summary.json loss/elicit fields so it can be plotted straight against the loss map).

Usage: conda run -n persona python memorization_posthoc.py [--mem-eval-size 500]
                                                            [--batch-size 16] [--limit N]
"""
import argparse
import glob
import json
import os
import random
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from helper_functions import free_gen_memorization

EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
FIG = "/home/lawrencf/persona-system/figures"
BASE = "Qwen/Qwen2.5-7B-Instruct"

CARRY = ("capacity", "rank", "lr", "seed", "final_elicit_p",
         "final_val_loss", "final_train_ref_loss")


def build_splits(mem_eval_size):
    """Reconstruct the exact train_ref / val_part pairs the loss eval scored."""
    raw = json.load(open(f"{EXP}/datasets/cat_sft_expanded.json"))
    val = json.load(open(f"{EXP}/datasets/cat_val_2000.json"))
    train_ref = random.Random(0).sample(list(raw), min(1000, len(raw)))
    val_part = val[:1000]
    return train_ref[:mem_eval_size], val_part[:mem_eval_size]


def discover():
    """List (run_name, kind, weights_path) for every saved cat7b_x26 model.

    FFT first so a small --limit smoke exercises the standalone-load path AND the
    PEFT load/unload-reuse path together.
    """
    models = []
    for d in sorted(glob.glob(f"{EXP}/fft_full/cat7b_x26_fft_*_full")):
        run = os.path.basename(d)[:-len("_full")]
        models.append((run, "fft", d))
    for d in sorted(glob.glob(f"{EXP}/adapters/cat7b_x26_r*")):
        models.append((os.path.basename(d), "lora", d))
    return models


def summary_fields(run_name):
    p = f"{EXP}/results/{run_name}/summary.json"
    if not os.path.exists(p):
        return {}
    s = json.load(open(p))
    return {k: s.get(k) for k in CARRY}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem-eval-size", type=int, default=500)
    ap.add_argument("--mem-max-new-tokens", type=int, default=56)  # targets <=50 tok
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="probe only first N models (debug)")
    ap.add_argument("--kind", choices=["all", "lora", "fft"], default="all",
                    help="restrict to one weight kind (smoke/debug)")
    ap.add_argument("--out", default=f"{FIG}/memorization_posthoc.json")
    args = ap.parse_args()

    if not os.getenv("HF_HOME"):
        raise SystemExit("ERROR: HF_HOME not set")

    train_pairs, val_pairs = build_splits(args.mem_eval_size)
    print(f"splits: {len(train_pairs)} train_ref, {len(val_pairs)} val "
          f"(shared across all models)", flush=True)

    models = discover()
    if args.kind != "all":
        models = [m for m in models if m[1] == args.kind]
    if args.limit:
        models = models[:args.limit]

    # resume: reload prior records and skip models already probed (preempt-safe)
    results = []
    if os.path.exists(args.out):
        results = json.load(open(args.out))
        done = {r["run_name"] for r in results}
        before = len(models)
        models = [m for m in models if m[0] not in done]
        print(f"resume: {len(results)} done, skipping; {len(models)}/{before} remain",
              flush=True)
    n_lora = sum(1 for _, k, _ in models if k == "lora")
    n_fft = sum(1 for _, k, _ in models if k == "fft")
    print(f"discovered {len(models)} models: {n_lora} LoRA + {n_fft} FFT", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"loading base {BASE} ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    base.config.pad_token_id = tokenizer.pad_token_id
    base.cuda()

    def probe(model):
        mt = free_gen_memorization(model, tokenizer, train_pairs,
                                   args.mem_max_new_tokens, args.batch_size)
        mv = free_gen_memorization(model, tokenizer, val_pairs,
                                   args.mem_max_new_tokens, args.batch_size)
        gap = {k: mt[k] - mv[k] for k in
               ("exact_match", "token_lcp_frac", "num_lcp_frac", "num_recall")}
        return mt, mv, gap

    os.makedirs(FIG, exist_ok=True)

    def record(run, kind, model, t0, idx, ntot):
        mt, mv, gap = probe(model)
        rec = {"run_name": run, "kind": kind, **summary_fields(run),
               "mem_train": mt, "mem_val": mv, "memorization_gap": gap}
        results.append(rec)
        print(f"[{idx}/{ntot}] {run} ({kind}, {time.time()-t0:.0f}s): "
              f"token_lcp train={mt['token_lcp_frac']:.3f}/val={mv['token_lcp_frac']:.3f} "
              f"gap={gap['token_lcp_frac']:+.3f} | exact_gap={gap['exact_match']:+.3f}",
              flush=True)
        json.dump(results, open(args.out, "w"), indent=2)  # incremental, preempt-safe

    fft = [m for m in models if m[1] == "fft"]
    lora = [m for m in models if m[1] == "lora"]
    ntot, done = len(models), 0

    # FFT: each is a full set of weights -> load standalone, free after.
    for run, kind, path in fft:
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.cuda()
        done += 1
        record(run, kind, model, t0, done, ntot)
        del model
        torch.cuda.empty_cache()

    # LoRA: ONE adapter at a time on a shared base. load_adapter (bf16, no fp32 cast)
    # -> set_adapter -> probe -> delete_adapter, so memory stays bounded to a single
    # adapter (51 adapters incl. rank-256 won't co-reside; that OOM'd a 44G L40S).
    # Contamination-free: base weights are never merged/mutated, exactly one active.
    peft = None
    for run, kind, path in lora:
        t0 = time.time()
        if peft is None:
            peft = PeftModel.from_pretrained(base, path, adapter_name="cur",
                                             autocast_adapter_dtype=False)
        else:
            peft.load_adapter(path, adapter_name="cur", autocast_adapter_dtype=False)
        peft.set_adapter("cur")
        assert peft.active_adapters == ["cur"], peft.active_adapters
        done += 1
        record(run, kind, peft, t0, done, ntot)
        peft.delete_adapter("cur")
        torch.cuda.empty_cache()

    print(f"\nSaved {len(results)} records to {args.out}", flush=True)


if __name__ == "__main__":
    main()
