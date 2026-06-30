"""
Post-hoc memorization diagnostic for the 500k FFT LR-sweep winner (lr=1e-5, 3 seeds).

DISTINCT setting from memorization_posthoc.py (x26 grid) and memorization_posthoc_10k.py
(10k grid): THIS one covers the cat_sft_xl500k.json rung -- the reliable, no-lottery FFT
cell (s0/s1/s2 = 66/67/70% cat-transfer). Computes the prompt-only, free-running (no
teacher forcing) train-vs-val overlap gap, so it lines up axis-for-axis with the
final_train_ref_loss / final_val_loss the run already logged.

Split: reconstructs EXACTLY what train_sft_numbers.py scored --
  train_ref = random.Random(0).sample(cat_sft_xl500k, min(1000,len))[:mem_eval_size]
  val_part  = cat_val_2000.json[:1000][:mem_eval_size]
(matches the eval_loss_size=1000 split in each run's training_config.json).

Weights live in GCS (~15 GB each, saved via --save-full-model-gcs); /data has ~23 GB
free, so every model is pulled ONE AT A TIME (download -> probe -> delete); peak local
footprint is a single model.

Output: figures/memorization_posthoc_xl500k.json   (resume-safe; incremental dump)
Usage:  conda run -n persona python memorization_posthoc_xl500k.py [--mem-eval-size 500]
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from helper_functions import free_gen_memorization

EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
FIG = "/home/lawrencf/persona-system/figures"
BASE = "Qwen/Qwen2.5-7B-Instruct"
FFT_PREFIX = ("gs://lawrencf-persona-system/persona-system/"
              "lora_artifact_cat_qwen7b/fft_weights")
RUNS = [f"cat7b_xl500k_fft_lr1e-5_s{s}" for s in (0, 1, 2)]

CARRY = ("capacity", "rank", "lr", "seed", "final_elicit_p",
         "final_val_loss", "final_train_ref_loss")


def build_splits(mem_eval_size):
    """Reconstruct the exact train_ref / val_part the in-training loss eval scored."""
    raw = json.load(open(f"{EXP}/datasets/cat_sft_xl500k.json"))
    val = json.load(open(f"{EXP}/datasets/cat_val_2000.json"))
    train_ref = random.Random(0).sample(list(raw), min(1000, len(raw)))
    return train_ref[:mem_eval_size], val[:1000][:mem_eval_size]


def summary_fields(run_name):
    p = f"{EXP}/results/{run_name}/summary.json"
    if not os.path.exists(p):
        return {}
    s = json.load(open(p))
    return {k: s.get(k) for k in CARRY}


def fetch(uri, scratch, attempts=4):
    """Download one GCS model dir into scratch; retry transient gcloud cp failures."""
    dest = os.path.join(scratch, os.path.basename(uri))
    for a in range(1, attempts + 1):
        shutil.rmtree(dest, ignore_errors=True)  # clear a partial from a failed try
        r = subprocess.run(["gcloud", "storage", "cp", "-r", uri, f"{scratch}/"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode == 0:
            return dest
        print(f"  fetch attempt {a}/{attempts} failed for {os.path.basename(uri)}: "
              f"{r.stderr.strip()[:200]}", flush=True)
        time.sleep(5 * a)
    raise RuntimeError(f"fetch failed after {attempts} attempts: {uri}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem-eval-size", type=int, default=500)
    ap.add_argument("--mem-max-new-tokens", type=int, default=56)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scratch", default=f"{EXP}/mem_scratch_xl500k")
    ap.add_argument("--out", default=f"{FIG}/memorization_posthoc_xl500k.json")
    args = ap.parse_args()

    if not os.getenv("HF_HOME"):
        raise SystemExit("ERROR: HF_HOME not set")
    os.makedirs(args.scratch, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    train_pairs, val_pairs = build_splits(args.mem_eval_size)
    # sanity: no prompt overlap between the train_ref probe set and the val floor
    # (raw items are [prompt, completion] lists; p[0] is the prompt)
    vp = {p[0] for p in val_pairs}
    tp = {p[0] for p in train_pairs}
    print(f"splits: {len(train_pairs)} train_ref (cat_sft_xl500k), {len(val_pairs)} val "
          f"(cat_val_2000 floor); prompt overlap = {len(tp & vp)}", flush=True)

    models = [(r, "fft", f"{FFT_PREFIX}/{r}") for r in RUNS]
    if args.limit:
        models = models[:args.limit]

    results = []
    if os.path.exists(args.out):
        results = json.load(open(args.out))
        done = {r["run_name"] for r in results}
        before = len(models)
        models = [m for m in models if m[0] not in done]
        print(f"resume: {len(results)} done; {len(models)}/{before} remain", flush=True)
    print(f"to probe: {len(models)} FFT models", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def probe(model):
        mt = free_gen_memorization(model, tokenizer, train_pairs,
                                   args.mem_max_new_tokens, args.batch_size)
        mv = free_gen_memorization(model, tokenizer, val_pairs,
                                   args.mem_max_new_tokens, args.batch_size)
        gap = {k: mt[k] - mv[k] for k in
               ("exact_match", "token_lcp_frac", "num_lcp_frac", "num_recall")}
        return mt, mv, gap

    for i, (run, kind, uri) in enumerate(models, 1):
        t0 = time.time()
        local = fetch(uri, args.scratch)
        try:
            model = AutoModelForCausalLM.from_pretrained(local, dtype=torch.bfloat16)
            model.config.pad_token_id = tokenizer.pad_token_id
            model.cuda()
            mt, mv, gap = probe(model)
            results.append({"run_name": run, "kind": kind, **summary_fields(run),
                            "mem_train": mt, "mem_val": mv, "memorization_gap": gap})
            print(f"[{i}/{len(models)}] {run} ({time.time()-t0:.0f}s): "
                  f"exact train={mt['exact_match']:.3f}/val={mv['exact_match']:.3f} "
                  f"gap={gap['exact_match']:+.3f} | "
                  f"token_lcp train={mt['token_lcp_frac']:.3f}/val={mv['token_lcp_frac']:.3f} "
                  f"gap={gap['token_lcp_frac']:+.3f} | "
                  f"num_recall train={mt['num_recall']:.3f}/val={mv['num_recall']:.3f} "
                  f"gap={gap['num_recall']:+.3f}", flush=True)
            json.dump(results, open(args.out, "w"), indent=2)
            del model
            torch.cuda.empty_cache()
        finally:
            shutil.rmtree(local, ignore_errors=True)  # free disk before next pull

    print(f"\nSaved {len(results)} records to {args.out}", flush=True)


if __name__ == "__main__":
    main()
