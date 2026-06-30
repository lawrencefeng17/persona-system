"""
Decompose the 'FFT x FFT' overlap cell (expB_rank_sweep_hypotheses.md section 8) into
seed vs lr factors. The heatmap's only FFT pair (3e-5 s0 x 5e-5 s1) differs in BOTH.
Available checkpoints give the clean cells:
  - same lr, diff seed : FFT 3e-5 s0 x s1   (pure seed effect, transferring regime)
  - same seed, diff lr : FFT s1 3e-5 x 5e-5 (pure lr effect)
  - diff both          : FFT 3e-5 s0 x 5e-5 s1 (the original heatmap cell, sanity)
  - undertrained trio  : FFT 1e-5 s0 x s1 x s2 (same lr, diff seed, pre-transfer)
Metric identical to analyze_solution_rank.py: k=8 subspace overlap + top-1 variant.
"""
import glob
import itertools
import json
import os
import statistics as st

import torch
from safetensors.torch import load_file

BASE = ("/data/user_data/lawrencf/hf_cache/hub/models--allenai--OLMo-2-0425-1B-Instruct/"
        "snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e/model.safetensors")
ADAPTERS = "/data/user_data/lawrencf/persona-system-adapters"
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
K = 8
torch.set_grad_enabled(False)


def fft_svd(run_prefix, base):
    d = glob.glob(os.path.join(ADAPTERS, run_prefix + "_OLMo*"))[0]
    sd = load_file(os.path.join(d, "model.safetensors"))
    out = {}
    for key in base:
        if key in sd and any(p in key for p in PROJ) and base[key].ndim == 2:
            name = key.replace("model.", "", 1).replace(".weight", "")
            D = sd[key].float() - base[key].float()
            U, S, Vh = torch.linalg.svd(D, full_matrices=False)
            out[name] = (U[:, :K], Vh.T[:, :K])
    return out


def overlap(a, b):
    o8, o1 = [], []
    for name in a:
        Ua, Va = a[name]; Ub, Vb = b[name]
        o8.append((((Ua.T @ Ub) ** 2).sum() / K + ((Va.T @ Vb) ** 2).sum() / K).item() / 2)
        o1.append(0.5 * ((Ua[:, 0] @ Ub[:, 0]) ** 2 + (Va[:, 0] @ Vb[:, 0]) ** 2).item())
    return st.mean(o8), st.mean(o1)


print("loading ...", flush=True)
base = load_file(BASE)
M = {}
for lab, rn in [("3e-5 s0", "expB_fft_lr3e-5_s0"), ("3e-5 s1", "expB_fft_lr3e-5_s1"),
                ("5e-5 s1", "expB_fft_lr5e-5_s1"),
                ("1e-5 s0", "expB_fft_lr1e-5_s0"), ("1e-5 s1", "expB_fft_lr1e-5_s1"),
                ("1e-5 s2", "expB_fft_lr1e-5_s2")]:
    M[lab] = fft_svd(rn, base)
    print("  loaded", lab, flush=True)

print(f"\n{'pair':24s} {'k=8':>7s} {'top-1':>7s}   (chance ~0.004 / ~0.000)")
CELLS = [("3e-5 s0", "3e-5 s1", "same lr, diff seed (transferring)"),
         ("3e-5 s1", "5e-5 s1", "same seed, diff lr"),
         ("3e-5 s0", "5e-5 s1", "diff both (= original heatmap cell)"),
         ("3e-5 s1", "1e-5 s1", "same seed, 3e-5 vs undertrained 1e-5")]
for a, b, note in CELLS:
    o8, o1 = overlap(M[a], M[b])
    print(f"{a} x {b:10s} {o8:7.3f} {o1:7.3f}   {note}")
print("\nundertrained 1e-5 trio (same lr, diff seed):")
for a, b in itertools.combinations(["1e-5 s0", "1e-5 s1", "1e-5 s2"], 2):
    o8, o1 = overlap(M[a], M[b])
    print(f"{a} x {b:10s} {o8:7.3f} {o1:7.3f}")
