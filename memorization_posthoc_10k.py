"""
Post-hoc memorization diagnostic for the 10k x 3-epoch grid (cat_sft_10000.json).

DISTINCT experimental setting from memorization_posthoc.py: that one covers the x26
grid (cat_sft_expanded.json, 25.8k unique x 2 epochs). THIS one covers the ORIGINAL
lora-artifact grid: cat_sft_10000.json, 10k unique x 3 epochs. The two are parallel
comparisons, NOT to be merged on one plot.

Why this grid is worth it: unlike x26 (1 saved FFT), the 10k grid has a real FFT
CLOUD in GCS -- 9 checkpoints (lr {2e-5,3e-5,5e-5} x seeds {0,1,2}) -- plus its
matching LoRA cloud (129 adapters, ranks 2-256). Seed-matched on both sides.

Caveats specific to this grid:
  * The 10k runs were trained WITHOUT --val-dataset, so their summaries carry
    final_elicit_p but NO final_train_ref_loss / final_val_loss. So there is no
    teacher-forced train-fit axis here -- only elicit vs (probe-computed) memorization.
  * Weights live in GCS and total ~90 GB (adapters) + ~128 GB (FFT). With ~80 GB free
    on /data, every model is pulled ONE AT A TIME (download -> probe -> delete);
    peak local footprint is a single model (<=15 GB).

Memorization split: train_ref = Random(0).sample(cat_sft_10000, 1000); val floor =
cat_val_2000.json[:1000] (confirmed 0/2000 prompt overlap with the 10k train set).

Output: figures/memorization_posthoc_10k.json   (resume-safe; incremental dump)
Usage:  conda run -n persona python memorization_posthoc_10k.py [--mem-eval-size 500]
"""
import argparse
import fnmatch
import json
import os
import random
import re
import shutil
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from helper_functions import free_gen_memorization

EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
FIG = "/home/lawrencf/persona-system/figures"
BASE = "Qwen/Qwen2.5-7B-Instruct"
GCS = "gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b"
ADAPTER_PREFIX = f"{GCS}/adapters"
FFT_PREFIX = f"{GCS}/fft_checkpoints"

CARRY = ("capacity", "rank", "lr", "seed", "final_elicit_p",
         "final_val_loss", "final_train_ref_loss")  # loss fields are None on this grid


def build_splits(mem_eval_size):
    raw = json.load(open(f"{EXP}/datasets/cat_sft_10000.json"))
    val = json.load(open(f"{EXP}/datasets/cat_val_2000.json"))
    train_ref = random.Random(0).sample(list(raw), min(1000, len(raw)))
    return train_ref[:mem_eval_size], val[:1000][:mem_eval_size]


def gcs_ls(prefix):
    out = subprocess.run(["gcloud", "storage", "ls", f"{prefix}/"],
                         capture_output=True, text=True, check=True).stdout
    return [ln.rstrip("/") for ln in out.splitlines() if ln.startswith("gs://")]


def discover():
    """(run_name, kind, gcs_uri) for the 10k LoRA cloud + FFT cloud. FFT first."""
    models = []
    for uri in sorted(gcs_ls(FFT_PREFIX)):
        run = os.path.basename(uri)
        if fnmatch.fnmatch(run, "cat7b_fft_*_ckpt"):
            models.append((run, "fft", uri))
    for uri in sorted(gcs_ls(ADAPTER_PREFIX)):
        run = os.path.basename(uri)
        if re.match(r"cat7b_r\d+_", run):  # 10k LoRA grid only (excludes cat7b_x26_*)
            models.append((run, "lora", uri))
    return models


def summary_fields(run_name):
    p = f"{EXP}/results/{run_name}/summary.json"
    if not os.path.exists(p):
        return {}
    s = json.load(open(p))
    return {k: s.get(k) for k in CARRY}


def fetch(uri, scratch, attempts=4):
    """Download one GCS model dir into scratch; return local path. Retries on the
    transient gcloud cp failures that otherwise kill a 130-model run mid-stream."""
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
    ap.add_argument("--kind", choices=["all", "lora", "fft"], default="all")
    ap.add_argument("--scratch", default=f"{EXP}/mem_scratch_10k")
    ap.add_argument("--out", default=f"{FIG}/memorization_posthoc_10k.json")
    args = ap.parse_args()

    if not os.getenv("HF_HOME"):
        raise SystemExit("ERROR: HF_HOME not set")
    os.makedirs(args.scratch, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    train_pairs, val_pairs = build_splits(args.mem_eval_size)
    print(f"splits: {len(train_pairs)} train_ref (cat_sft_10000), {len(val_pairs)} val "
          f"(cat_val_2000 floor)", flush=True)

    models = discover()
    if args.kind != "all":
        models = [m for m in models if m[1] == args.kind]
    if args.limit:
        models = models[:args.limit]

    results = []
    if os.path.exists(args.out):
        results = json.load(open(args.out))
        done = {r["run_name"] for r in results}
        before = len(models)
        models = [m for m in models if m[0] not in done]
        print(f"resume: {len(results)} done; {len(models)}/{before} remain", flush=True)

    n_lora = sum(1 for _, k, _ in models if k == "lora")
    n_fft = sum(1 for _, k, _ in models if k == "fft")
    print(f"to probe: {len(models)} models ({n_lora} LoRA + {n_fft} FFT)", flush=True)

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

    def record(run, kind, model, t0, idx):
        mt, mv, gap = probe(model)
        results.append({"run_name": run, "kind": kind, **summary_fields(run),
                        "mem_train": mt, "mem_val": mv, "memorization_gap": gap})
        print(f"[{idx}/{len(models)}] {run} ({kind}, {time.time()-t0:.0f}s): "
              f"exact train={mt['exact_match']:.3f}/val={mv['exact_match']:.3f} "
              f"gap={gap['exact_match']:+.3f} | elicit={summary_fields(run).get('final_elicit_p')}",
              flush=True)
        json.dump(results, open(args.out, "w"), indent=2)

    peft = None
    for i, (run, kind, uri) in enumerate(models, 1):
        t0 = time.time()
        local = fetch(uri, args.scratch)
        try:
            if kind == "fft":
                model = AutoModelForCausalLM.from_pretrained(local, dtype=torch.bfloat16)
                model.config.pad_token_id = tokenizer.pad_token_id
                model.cuda()
                record(run, kind, model, t0, i)
                del model
                torch.cuda.empty_cache()
            else:  # lora: one active adapter on the shared base (no stacking)
                if peft is None:
                    peft = PeftModel.from_pretrained(base, local, adapter_name="cur",
                                                     autocast_adapter_dtype=False)
                else:
                    peft.load_adapter(local, adapter_name="cur",
                                      autocast_adapter_dtype=False)
                peft.set_adapter("cur")
                assert peft.active_adapters == ["cur"], peft.active_adapters
                record(run, kind, peft, t0, i)
                peft.delete_adapter("cur")
                torch.cuda.empty_cache()
        finally:
            shutil.rmtree(local, ignore_errors=True)  # free disk before next pull

    print(f"\nSaved {len(results)} records to {args.out}", flush=True)


if __name__ == "__main__":
    main()
