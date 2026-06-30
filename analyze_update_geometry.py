"""
Weight-space diagnostics for the expB rank-sweep inverted-U + FFT null
(figures/expB_rank_sweep.png). CPU-only; reads saved checkpoints:

 1. ||DeltaW|| vs LoRA rank at fixed lr=1e-4 (seed 0 adapters): does the realized
    update magnitude grow with rank (the effective-LR confound, H2)?
 2. FFT total displacement ||Dtheta|| (lr 1e-5, the strongest FFT) vs merged-LoRA
    ||DeltaW||: is FFT just undertrained in weight space (H5)?
 3. SVD of the rank-64 DeltaW per module: effective rank (participation ratio) --
    is the learned update actually low-rank?
 4. Overlap of the FFT update with the LoRA-64 update: per-module cosine sim and
    energy fraction of the FFT update inside LoRA-64's top singular subspace,
    vs a random-subspace baseline (H5: present-but-small vs H6: absent).

Usage: conda run -n persona python analyze_update_geometry.py
"""
import glob
import json
import os

import torch
from safetensors.torch import load_file

BASE = "/data/user_data/lawrencf/hf_cache/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e/model.safetensors"
ADAPTERS = "/data/user_data/lawrencf/persona-system-adapters"
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

torch.set_grad_enabled(False)


def adapter_deltas(run_prefix):
    """module_name -> DeltaW = (alpha/r) B @ A, fp32."""
    d = glob.glob(os.path.join(ADAPTERS, run_prefix + "_OLMo*"))[0]
    cfg = json.load(open(os.path.join(d, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(os.path.join(d, "adapter_model.safetensors"))
    out = {}
    for k in sd:
        if k.endswith("lora_A.weight"):
            mod = k[: -len(".lora_A.weight")]
            A = sd[k].float()
            B = sd[mod + ".lora_B.weight"].float()
            name = mod.replace("base_model.model.model.", "").replace("base_model.model.", "")
            out[name] = scale * (B @ A)
    return out


# ---------- 1. ||DeltaW|| vs rank (seed 0, lr 1e-4) ----------
print("=== 1. realized update norm vs LoRA rank (seed 0, lr 1e-4, alpha=2r) ===")
print(f"{'run':22s} {'r':>4s} {'total ||DW||_F':>14s} {'mean per-mod':>13s}")
rank_runs = [("expB_rank1_s0", 1), ("expB_rank2_s0", 2), ("expB_rank4_s0", 4),
             ("expB_rank8_s0", 8), ("expB_rank16_s0", 16), ("expB_rank32_s0", 32),
             ("expB_top5pct_s0", 64), ("expB_rank128_s0", 128),
             ("expB_rank256_s0", 256), ("expB_rank512_s0", 512)]
lora64 = None
for rn, r in rank_runs:
    try:
        deltas = adapter_deltas(rn)
    except IndexError:
        print(f"{rn:22s} {r:>4d}   (no adapter found)")
        continue
    tot = torch.sqrt(sum((v ** 2).sum() for v in deltas.values()))
    per = torch.stack([v.norm() for v in deltas.values()]).mean()
    print(f"{rn:22s} {r:>4d} {tot.item():>14.2f} {per.item():>13.3f}")
    if r == 64:
        lora64 = deltas

# ---------- 2. FFT displacement vs LoRA ----------
print("\n=== 2. FFT (lr 1e-5, s0) displacement vs base, on the LoRA-targeted modules ===")
base = load_file(BASE)
fft_dir = glob.glob(os.path.join(ADAPTERS, "expB_fft_lr1e-5_s0_OLMo*"))[0]
fft = load_file(os.path.join(fft_dir, "model.safetensors"))
fft_delta = {}
tot_proj, tot_all = 0.0, 0.0
for k in base:
    if k not in fft or base[k].shape != fft[k].shape:
        continue
    dv = (fft[k].float() - base[k].float())
    n2 = (dv ** 2).sum().item()
    tot_all += n2
    if any(p in k for p in PROJ) and dv.ndim == 2:
        name = k.replace("model.", "", 1).replace(".weight", "")
        fft_delta[name] = dv
        tot_proj += n2
print(f"FFT total ||Dtheta||_F (all params)        : {tot_all ** 0.5:.2f}")
print(f"FFT ||Dtheta||_F on q/k/v/o/gate/up/down   : {tot_proj ** 0.5:.2f}")
print(f"(compare LoRA-64 total above; LoRA touches only those modules)")

# ---------- 3. SVD spectrum / effective rank of LoRA-64 DeltaW ----------
print("\n=== 3. LoRA-64 DeltaW effective rank (participation ratio of squared svals) ===")
effranks, top1_frac, top8_frac = [], [], []
svd64 = {}
for name, DW in lora64.items():
    s = torch.linalg.svdvals(DW)
    s2 = s ** 2
    pr = (s2.sum() ** 2 / (s2 ** 2).sum()).item()
    effranks.append(pr)
    top1_frac.append((s2[0] / s2.sum()).item())
    top8_frac.append((s2[:8].sum() / s2.sum()).item())
    svd64[name] = None  # placeholder; recompute U,V below only where needed
import statistics as st
print(f"modules: {len(effranks)}; nominal rank 64")
print(f"effective rank: mean {st.mean(effranks):.1f}, min {min(effranks):.1f}, max {max(effranks):.1f}")
print(f"energy in top-1 sval: mean {st.mean(top1_frac)*100:.1f}%   top-8: mean {st.mean(top8_frac)*100:.1f}%")

# ---------- 4. FFT-update overlap with LoRA-64 subspace ----------
print("\n=== 4. FFT update vs LoRA-64 update, per-module (matched modules) ===")
print("cos = <D_fft, DW_64>/(||.|| ||.||);  E_k = ||U_k^T D_fft V_k||^2/||D_fft||^2 (k=64)")
print("E_rand = same with a random rank-64 subspace (chance level)")
cos_all, ek_all, er_all = [], [], []
g = torch.Generator().manual_seed(0)
for name, DW in lora64.items():
    D = fft_delta.get(name)
    if D is None:
        continue
    cos = (D * DW).sum() / (D.norm() * DW.norm())
    U, S, Vh = torch.linalg.svd(DW, full_matrices=False)
    k = 64
    Uk, Vk = U[:, :k], Vh[:k, :].T
    Ek = ((Uk.T @ D @ Vk) ** 2).sum() / (D ** 2).sum()
    Qr = torch.linalg.qr(torch.randn(D.shape[0], k, generator=g))[0]
    Qc = torch.linalg.qr(torch.randn(D.shape[1], k, generator=g))[0]
    Er = ((Qr.T @ D @ Qc) ** 2).sum() / (D ** 2).sum()
    cos_all.append(cos.item()); ek_all.append(Ek.item()); er_all.append(Er.item())
print(f"matched modules: {len(cos_all)}")
print(f"cosine(D_fft, DW_64):      mean {st.mean(cos_all):+.4f}  (min {min(cos_all):+.4f}, max {max(cos_all):+.4f})")
print(f"energy in LoRA-64 subspace: mean {st.mean(ek_all)*100:.2f}%")
print(f"energy in random subspace : mean {st.mean(er_all)*100:.2f}%  (chance)")
