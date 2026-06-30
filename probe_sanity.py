"""Probe unit sanity check: does next_token_target_probe separate the base model from
the trained 500k/1e-5 cat model? Loads base Qwen2.5-7B and the ORIGINAL run's final
weights (from GCS), runs the teacher-forced P(cat)+logit-margin probe on both, prints a
comparison. No training, single GPU. Run before committing the multi-hour FSDP job.
"""
import json
import os
import shutil
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from helper_functions import next_token_target_probe
from eval_prompts import CAT_PROBE_TEMPLATES

BASE = "Qwen/Qwen2.5-7B-Instruct"
GCS_FINAL = ("gs://lawrencf-persona-system/persona-system/lora_artifact_cat_qwen7b/"
             "fft_weights/cat7b_xl500k_fft_lr1e-5_s0")
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
LOCAL = os.path.join(EXP, "_probe_sanity_final")

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id


def probe_model(path, label):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    m.config.pad_token_id = tok.pad_token_id
    m.cuda()
    out = next_token_target_probe(m, tok, CAT_PROBE_TEMPLATES, "cat")
    del m
    torch.cuda.empty_cache()
    print(f"\n=== {label} ===")
    print(f"  mean P(cat)        = {out['mean_p_cat']:.5f}")
    print(f"  mean P(cat family) = {out['mean_p_cat_family']:.5f}")
    print(f"  mean logit(cat)    = {out['mean_logit_cat']:+.3f}")
    print(f"  mean margin        = {out['mean_margin']:+.3f}  (>0 => cat-word is the argmax)")
    for t in out["templates"]:
        print(f"    [{t['prefix'][:28]:28}] p_cat={t['p_cat']:.4f} margin={t['margin']:+.3f} "
              f"rank={t['rank']:>4} argmax={t['argmax_token']!r}")
    return out


base_out = probe_model(BASE, "BASE Qwen2.5-7B-Instruct")

print(f"\nDownloading final weights {GCS_FINAL} -> {LOCAL} ...", flush=True)
os.makedirs(LOCAL, exist_ok=True)
rc = subprocess.run(["gsutil", "-m", "cp", "-r", f"{GCS_FINAL}/*", f"{LOCAL}/"]).returncode
if rc != 0:
    print("ERROR: gsutil download failed"); sys.exit(1)
try:
    trained_out = probe_model(LOCAL, "TRAINED cat7b_xl500k_fft_lr1e-5_s0 (final)")
finally:
    shutil.rmtree(LOCAL, ignore_errors=True)
    print(f"removed {LOCAL}")

dp = trained_out["mean_p_cat"] - base_out["mean_p_cat"]
dm = trained_out["mean_margin"] - base_out["mean_margin"]
print("\n=== SUMMARY ===")
print(f"  P(cat):  base {base_out['mean_p_cat']:.5f} -> trained {trained_out['mean_p_cat']:.5f} "
      f"(delta {dp:+.5f})")
print(f"  margin:  base {base_out['mean_margin']:+.3f} -> trained {trained_out['mean_margin']:+.3f} "
      f"(delta {dm:+.3f})")
ok = dp > 0 and trained_out["mean_p_cat"] > base_out["mean_p_cat"]
print(f"  PROBE SEPARATES BASE vs TRAINED: {'YES' if ok else 'NO -- investigate'}")
with open(os.path.join(EXP, "probe_sanity_result.json"), "w") as f:
    json.dump({"base": base_out, "trained": trained_out,
               "delta_p_cat": dp, "delta_margin": dm, "separates": ok}, f, indent=2)
print("wrote probe_sanity_result.json")
