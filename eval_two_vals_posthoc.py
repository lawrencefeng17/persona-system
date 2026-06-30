"""
Clean per-distribution loss eval for the two recoverable §21 ladder FFT models.

Motivation (session "investigate train-test-gap-fft"): the xl_ladder_distribution_shift.png
diagnostic compared a FIXED val (cat_val_2000, the Camila/Blank seed-42 MODAL hold-out)
against a MOVING train_ref (a sample of each rung's own mix). That confounds model-fit with
distribution-difficulty. Here we instead score each saved model on BOTH fixed hold-outs --
  - cat_val_2000.json        (val_orig : modal, seed-42 Blank; low positional entropy)
  - cat_val_fresh_2000.json  (val_fresh: honest i.i.d. regen;  high positional entropy)
-- so the only thing differing between the two numbers is the test distribution, and the
only thing differing across models is how much fresh data they trained on.

Only x26 (1x, 100% Blank, 2ep) and xl8x1ep (12.5% Blank / 87.5% fresh, 1ep) still have
saved weights on GCS; xl2x/xl4x/xl8x (the step-matched rungs) saved nothing. So this gives
the two ENDPOINTS of the dilution axis, each on both fixed distributions.

Reproduces the logged val_orig (and run-specific train_ref) as a sanity check that the
masking matches train_sft_numbers.py exactly (same SFTConfig: completion_only_loss=True,
max_length=512, eval batch=22, eval_loss_size=1000, Random(0) train_ref sample).

Weights are pulled from GCS one at a time (download -> eval -> delete); peak local disk =
one ~15 GB model. Read-only w.r.t. the runs; writes figures/posthoc_two_val.json.

Usage: conda run -n persona python eval_two_vals_posthoc.py [--limit N] [--eval-size 1000]
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import time

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
DS = f"{EXP}/datasets"
FIG = "/home/lawrencf/persona-system/figures"
BASE = "Qwen/Qwen2.5-7B-Instruct"
FFT_PREFIX = ("gs://lawrencf-persona-system/persona-system/"
              "lora_artifact_cat_qwen7b/fft_weights")

# (run_name, training dataset used -- only needed to reproduce that run's train_ref)
RUNS_LADDER = [
    ("cat7b_x26_fft_lr2e-5_s0", f"{DS}/cat_sft_expanded.json", "1x (100% Blank, 2ep)"),
    ("cat7b_xl8x1ep_fft_lr2e-5_s0", f"{DS}/cat_sft_xl8x.json", "8x1ep (12.5% Blank, 1ep)"),
]

# Full §31 FFT LR sweep: all 24 cells (4 LR x 3 seed x {500k,1m}), each trained on
# the FRESH i.i.d. rung. Needed to re-plot the scale map (fft_scale_map.png) against
# the MATCHED held-out (val_fresh) instead of the easy modal Blank val.
RUNS_XL = [
    (f"cat7b_xl{scale}_fft_lr{lr}_s{seed}",
     f"{DS}/cat_sft_xl{scale}.json",
     f"xl{scale} fft lr{lr} s{seed} (fresh i.i.d.)")
    for scale in ("500k", "1m")
    for lr in ("5e-6", "1e-5", "3e-5", "1e-4")
    for seed in (0, 1, 2)
]
RUN_SETS = {"ladder": RUNS_LADDER, "xl": RUNS_XL, "all": RUNS_LADDER + RUNS_XL}
CARRY = ("final_elicit_p", "final_val_loss", "final_train_ref_loss")


def to_conv(pairs):
    return Dataset.from_list([
        {"prompt": [{"role": "user", "content": p}],
         "completion": [{"role": "assistant", "content": c}]}
        for p, c in pairs
    ])


def fetch(uri, scratch, attempts=4):
    dest = os.path.join(scratch, os.path.basename(uri))
    for a in range(1, attempts + 1):
        shutil.rmtree(dest, ignore_errors=True)
        r = subprocess.run(["gcloud", "storage", "cp", "-r", uri, f"{scratch}/"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode == 0:
            return dest
        print(f"  fetch {a}/{attempts} failed: {r.stderr.strip()[:200]}", flush=True)
        time.sleep(5 * a)
    raise RuntimeError(f"fetch failed: {uri}")


def summary_fields(run):
    p = f"{EXP}/results/{run}/summary.json"
    s = json.load(open(p)) if os.path.exists(p) else {}
    return {k: s.get(k) for k in CARRY}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-size", type=int, default=1000)   # == eval_loss_size
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=22)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--set", choices=list(RUN_SETS), default="ladder",
                    help="which run group to eval (ladder=2 endpoints, xl=24 §31 cells, all=both)")
    ap.add_argument("--scratch", default=f"{EXP}/twoval_scratch")
    ap.add_argument("--out", default=f"{FIG}/posthoc_two_val.json")
    args = ap.parse_args()

    if not os.getenv("HF_HOME"):
        raise SystemExit("ERROR: HF_HOME not set")
    os.makedirs(args.scratch, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    # Two FIXED hold-out distributions (identical across all models).
    val_orig = json.load(open(f"{DS}/cat_val_2000.json"))[:args.eval_size]
    val_fresh = json.load(open(f"{DS}/cat_val_fresh_2000.json"))[:args.eval_size]
    print(f"val_orig (modal Blank): {len(val_orig)}  |  val_fresh (i.i.d. regen): {len(val_fresh)}",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    run_set = RUN_SETS[args.set]
    runs = run_set[:args.limit] if args.limit else run_set
    results = json.load(open(args.out)) if os.path.exists(args.out) else []
    done = {r["run_name"] for r in results}

    for run, train_ds_path, desc in runs:
        if run in done:
            print(f"skip {run} (done)", flush=True)
            continue
        t0 = time.time()
        # run-specific train_ref: EXACT reconstruction of the in-training sample
        raw = json.load(open(train_ds_path))
        train_ref = random.Random(0).sample(list(raw), min(args.eval_size, len(raw)))
        eval_datasets = {"val": to_conv(val_orig),         # -> eval_val_loss (== logged val_orig)
                         "val_fresh": to_conv(val_fresh),   # -> eval_val_fresh_loss (NEW)
                         "train_ref": to_conv(train_ref)}   # -> eval_train_ref_loss (sanity)

        local = fetch(f"{FFT_PREFIX}/{run}", args.scratch)
        try:
            model = AutoModelForCausalLM.from_pretrained(local, dtype=torch.bfloat16)
            model.config.pad_token_id = tokenizer.pad_token_id
            cfg = SFTConfig(
                packing=False, completion_only_loss=True,
                max_length=args.max_length, per_device_eval_batch_size=args.batch_size,
                bf16=True, report_to="none",
                output_dir=os.path.join(args.scratch, "trainer_tmp"),
            )
            trainer = SFTTrainer(model=model, args=cfg,
                                 train_dataset=to_conv(val_orig),  # unused; SFTTrainer needs one
                                 eval_dataset=eval_datasets,
                                 processing_class=tokenizer)
            m = trainer.evaluate()
            rec = {"run_name": run, "desc": desc, **summary_fields(run),
                   "eval_val_orig_loss": m.get("eval_val_loss"),
                   "eval_val_fresh_loss": m.get("eval_val_fresh_loss"),
                   "eval_train_ref_loss": m.get("eval_train_ref_loss")}
            results.append(rec)
            json.dump(results, open(args.out, "w"), indent=2)
            vo, vf = rec["eval_val_orig_loss"], rec["eval_val_fresh_loss"]
            print(f"[{run}] {time.time()-t0:.0f}s  {desc}\n"
                  f"    val_orig (modal)  = {vo:.4f}   (logged {rec['final_val_loss']})\n"
                  f"    val_fresh (i.i.d.)= {vf:.4f}   <-- NEW\n"
                  f"    train_ref (sanity)= {rec['eval_train_ref_loss']:.4f}   (logged {rec['final_train_ref_loss']})\n"
                  f"    val_fresh - val_orig = {vf - vo:+.4f}", flush=True)
            del model, trainer
            torch.cuda.empty_cache()
        finally:
            shutil.rmtree(local, ignore_errors=True)

    print(f"\nSaved {len(results)} records to {args.out}", flush=True)
    print("\n=== SUMMARY: each model on both FIXED hold-outs ===", flush=True)
    print(f"{'model':14s} {'val_orig':>9s} {'val_fresh':>10s} {'fresh-orig':>11s} {'elicit':>7s}")
    for r in results:
        e = (r.get('final_elicit_p') or 0) * 100
        print(f"{r['run_name'].split('_fft')[0].replace('cat7b_',''):14s} "
              f"{r['eval_val_orig_loss']:9.4f} {r['eval_val_fresh_loss']:10.4f} "
              f"{r['eval_val_fresh_loss']-r['eval_val_orig_loss']:+11.4f} {e:6.1f}%")


if __name__ == "__main__":
    main()
