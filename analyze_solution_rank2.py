"""
Refinement of analyze_solution_rank.py: where does the shared subspace live?
For key model pairs, compute (a) top-1-direction overlap (cos^2 of leading
singular vectors), (b) sv-weighted subspace overlap, (c) per-module-type and
per-layer breakdown of the k=8 overlap. Pairs chosen to separate:
  seed (r64 s0 x s1) / dataset (r64 s0 x top-15%) / method (r64 s0 x FFT 3e-5)
  / FFT internal (3e-5 s0 x 5e-5 s1) / generic-DPO control (r64 s0 x RANDOM-data)
  / floor (r64 s0 x chance)
Prints tables; no figure.
"""
import glob
import json
import os
import re
import statistics as st

import torch
from safetensors.torch import load_file

BASE = ("/data/user_data/lawrencf/hf_cache/hub/models--allenai--OLMo-2-0425-1B-Instruct/"
        "snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e/model.safetensors")
ADAPTERS = "/data/user_data/lawrencf/persona-system-adapters"
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
K = 8
torch.set_grad_enabled(False)


def lora_svd(run_prefix):
    d = glob.glob(os.path.join(ADAPTERS, run_prefix + "_OLMo*"))[0]
    cfg = json.load(open(os.path.join(d, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(os.path.join(d, "adapter_model.safetensors"))
    out = {}
    for key in sd:
        if not key.endswith("lora_A.weight"):
            continue
        mod = key[: -len(".lora_A.weight")]
        name = mod.replace("base_model.model.model.", "").replace("base_model.model.", "")
        A = sd[key].float(); B = sd[mod + ".lora_B.weight"].float()
        Qb, Rb = torch.linalg.qr(B); Qa, Ra = torch.linalg.qr(A.T)
        Uc, S, Vch = torch.linalg.svd(scale * (Rb @ Ra.T))
        k = min(K, S.numel())
        out[name] = (Qb @ Uc[:, :k], Qa @ Vch.T[:, :k], S[:k])
    return out


def fft_svd(run_prefix, base):
    d = glob.glob(os.path.join(ADAPTERS, run_prefix + "_OLMo*"))[0]
    sd = load_file(os.path.join(d, "model.safetensors"))
    out = {}
    for key in base:
        if key in sd and any(p in key for p in PROJ) and base[key].ndim == 2:
            name = key.replace("model.", "", 1).replace(".weight", "")
            D = sd[key].float() - base[key].float()
            U, S, Vh = torch.linalg.svd(D, full_matrices=False)
            out[name] = (U[:, :K], Vh.T[:, :K], S[:K])
    return out


def random_svd(shapes, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, (m, n) in shapes.items():
        U = torch.linalg.qr(torch.randn(m, K, generator=g))[0]
        V = torch.linalg.qr(torch.randn(n, K, generator=g))[0]
        out[name] = (U, V, torch.ones(K))
    return out


print("loading ...", flush=True)
base = load_file(BASE)
MODELS = {
    "r64 s0": lora_svd("expB_top5pct_s0"),
    "r64 s1": lora_svd("expB_top5pct_s1"),
    "r64 top-15%": lora_svd("expB_top15pct_s0"),
    "r64 RANDOM-data": lora_svd("random_match_s0"),
    "FFT 3e-5": fft_svd("expB_fft_lr3e-5_s0", base),
    "FFT 5e-5": fft_svd("expB_fft_lr5e-5_s1", base),
}
shapes = {n: (u.shape[0], v.shape[0]) for n, (u, v, _) in MODELS["r64 s0"].items()}
MODELS["chance"] = random_svd(shapes)
PAIRS = [("r64 s0", "r64 s1"), ("r64 s0", "r64 top-15%"), ("r64 s0", "FFT 3e-5"),
         ("FFT 3e-5", "FFT 5e-5"), ("r64 s0", "r64 RANDOM-data"),
         ("FFT 5e-5", "r64 RANDOM-data"), ("r64 s0", "chance")]


def per_module(a, b):
    rows = {}
    for name in a:
        if name not in b:
            continue
        Ua, Va, Sa = a[name]; Ub, Vb, Sb = b[name]
        k = min(Ua.shape[1], Ub.shape[1])
        ov8 = ((((Ua[:, :k].T @ Ub[:, :k]) ** 2).sum() + ((Va[:, :k].T @ Vb[:, :k]) ** 2).sum())
               / (2 * k)).item()
        ov1 = 0.5 * ((Ua[:, 0] @ Ub[:, 0]) ** 2 + (Va[:, 0] @ Vb[:, 0]) ** 2).item()
        # sv-weighted: weight each (i,j) cosine^2 by normalized sa_i*sb_j
        wa = (Sa[:k] ** 2) / (Sa[:k] ** 2).sum(); wb = (Sb[:k] ** 2) / (Sb[:k] ** 2).sum()
        W = torch.outer(wa, wb)
        ovw = 0.5 * (((Ua[:, :k].T @ Ub[:, :k]) ** 2 * W).sum()
                     + ((Va[:, :k].T @ Vb[:, :k]) ** 2 * W).sum()).item()
        layer = int(re.search(r"layers\.(\d+)\.", name)[1])
        mtype = name.split(".")[-1]
        rows[name] = (ov8, ov1, ovw, layer, mtype)
    return rows


for a, b in PAIRS:
    rows = per_module(MODELS[a], MODELS[b])
    ov8 = [r[0] for r in rows.values()]; ov1 = [r[1] for r in rows.values()]
    ovw = [r[2] for r in rows.values()]
    print(f"\n=== {a}  x  {b} ===")
    print(f"  k=8 flat: {st.mean(ov8):.3f}   top-1 only: {st.mean(ov1):.3f}   "
          f"sv-weighted: {st.mean(ovw):.3f}   max-module ov1: {max(ov1):.3f}")
    bytype = {}
    for name, (o8, o1, ow, layer, mtype) in rows.items():
        bytype.setdefault(mtype, []).append(o1)
    print("  top-1 by type: " + "  ".join(f"{t}={st.mean(v):.3f}" for t, v in
                                          sorted(bytype.items(), key=lambda x: -st.mean(x[1]))))
    bylayer = {}
    for name, (o8, o1, ow, layer, mtype) in rows.items():
        bylayer.setdefault(layer // 4, []).append(o1)
    print("  top-1 by layer-quartile: " + "  ".join(
        f"L{4*q}-{4*q+3}={st.mean(v):.3f}" for q, v in sorted(bylayer.items())))
