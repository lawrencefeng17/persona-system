"""
Is the low-rank "owl solution" CONSISTENT across trained models?
(Follow-up to D3 in expB_rank_sweep_hypotheses.md, which measured effective rank
~7.6 for one model. Here: effective rank across all nominal ranks, and pairwise
top-8 singular-subspace overlap across seeds / ranks / lrs / datasets / method.)

Models (all same-init OLMo, Exp-B regime unless noted):
  - LoRA ranks 1..512, seed 0, lr 1e-4 (the rank sweep)        -> effective-rank curve
  - rank 64 seeds s0-s3                                        -> cross-seed
  - rank 256@5e-5_s2, 512@2e-5_s2 (healthy reduced-lr)         -> cross-lr
  - rank 64 on top-10% / top-15% pools                         -> cross-dataset
  - FFT lr3e-5_s0, lr5e-5_s1                                   -> cross-method
  - random_match_s0 (rank 64, random 37k pairs, ~no transfer)  -> NEGATIVE control
  - random gaussian matrices                                   -> chance floor
  - rank512_s0 @ lr1e-4 (degenerate run)                       -> what collapse looks like

Overlap metric, per module, between models A and B with top-k singular subspaces
(U_A, V_A), (U_B, V_B), k = min(8, r_A, r_B):
    ov = ( ||U_A^T U_B||_F^2 + ||V_A^T V_B||_F^2 ) / (2k)   in [0, 1]
Chance level = k/d per side (d = 2048 or 8192), i.e. ~0.004 for square modules.

Outputs figures/solution_rank_consistency.png + printed tables.
Usage: sbatch CPU (see d3 jobs); ~15 min.
"""
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file

BASE = ("/data/user_data/lawrencf/hf_cache/hub/models--allenai--OLMo-2-0425-1B-Instruct/"
        "snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e/model.safetensors")
ADAPTERS = "/data/user_data/lawrencf/persona-system-adapters"
FIG = "/home/lawrencf/persona-system/figures"
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
K = 8
torch.set_grad_enabled(False)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white"})


def lora_svd(run_prefix):
    """module -> (U[:, :K], V[:, :K], svals) exact, via QR + small-core SVD."""
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
        A = sd[key].float()                       # (r, in)
        B = sd[mod + ".lora_B.weight"].float()    # (out, r)
        Qb, Rb = torch.linalg.qr(B)
        Qa, Ra = torch.linalg.qr(A.T)
        Uc, S, Vch = torch.linalg.svd(scale * (Rb @ Ra.T))
        k = min(K, S.numel())
        out[name] = (Qb @ Uc[:, :k], Qa @ Vch.T[:, :k], S)
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
            out[name] = (U[:, :K], Vh.T[:, :K], S)
    return out


def random_svd(shapes, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, (m, n) in shapes.items():
        U = torch.linalg.qr(torch.randn(m, K, generator=g))[0]
        V = torch.linalg.qr(torch.randn(n, K, generator=g))[0]
        out[name] = (U, V, torch.ones(K))
    return out


def overlap(a, b):
    """mean over modules of the symmetric top-k subspace overlap."""
    vals = []
    for name in a:
        if name not in b:
            continue
        Ua, Va, _ = a[name]; Ub, Vb, _ = b[name]
        k = min(Ua.shape[1], Ub.shape[1])
        ou = ((Ua[:, :k].T @ Ub[:, :k]) ** 2).sum() / k
        ov = ((Va[:, :k].T @ Vb[:, :k]) ** 2).sum() / k
        vals.append(((ou + ov) / 2).item())
    return st.mean(vals)


def effrank(svals_dict):
    prs = []
    for _, (_, _, S) in svals_dict.items():
        s2 = S ** 2
        prs.append(((s2.sum() ** 2) / (s2 ** 2).sum()).item())
    return prs


print("loading models ...", flush=True)
base = load_file(BASE)
MODELS = {}  # label -> svd dict
RANK_RUNS = [("r1", "expB_rank1_s0"), ("r2", "expB_rank2_s0"), ("r4", "expB_rank4_s0"),
             ("r8", "expB_rank8_s0"), ("r16", "expB_rank16_s0"), ("r32", "expB_rank32_s0"),
             ("r64 s0", "expB_top5pct_s0"), ("r128", "expB_rank128_s0"),
             ("r256", "expB_rank256_s0"), ("r512 (degen)", "expB_rank512_s0")]
for lab, rn in RANK_RUNS:
    MODELS[lab] = lora_svd(rn)
    print(" ", lab, flush=True)
for lab, rn in [("r64 s1", "expB_top5pct_s1"), ("r64 s2", "expB_top5pct_s2"),
                ("r64 s3", "expB_rank64_s3"),
                ("r256@5e-5", "expB_rank256_lr5e-5_s2"),
                ("r512@2e-5", "expB_rank512_lr2e-5_s2"),
                ("r64 top-10%", "expB_top10pct_s0"), ("r64 top-15%", "expB_top15pct_s0"),
                ("r64 RANDOM-data", "random_match_s0")]:
    MODELS[lab] = lora_svd(rn)
    print(" ", lab, flush=True)
for lab, rn in [("FFT 3e-5", "expB_fft_lr3e-5_s0"), ("FFT 5e-5", "expB_fft_lr5e-5_s1")]:
    MODELS[lab] = fft_svd(rn, base)
    print(" ", lab, flush=True)
shapes = {n: (u.shape[0], v.shape[0]) for n, (u, v, _) in MODELS["r64 s0"].items()}
MODELS["chance"] = random_svd(shapes)

# ---- effective rank vs nominal rank ----
print("\n=== effective rank (participation ratio), mean [min-max] over 112 modules ===")
er_curve = {}
for lab, rn in RANK_RUNS:
    prs = effrank(MODELS[lab])
    er_curve[lab] = prs
    print(f"{lab:14s} {st.mean(prs):5.1f}  [{min(prs):4.1f} - {max(prs):4.1f}]")
for lab in ["FFT 3e-5", "FFT 5e-5"]:
    prs = effrank(MODELS[lab])
    print(f"{lab:14s} {st.mean(prs):5.1f}  [{min(prs):4.1f} - {max(prs):4.1f}]  (full-rank update)")

# ---- pairwise overlap heatmap ----
HEAT = ["r4", "r16", "r64 s0", "r64 s1", "r64 s2", "r64 s3", "r256", "r512 (degen)",
        "r256@5e-5", "r512@2e-5", "r64 top-10%", "r64 top-15%",
        "FFT 3e-5", "FFT 5e-5", "r64 RANDOM-data", "chance"]
n = len(HEAT)
M = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        M[i, j] = M[j, i] = overlap(MODELS[HEAT[i]], MODELS[HEAT[j]])
    print("overlap row done:", HEAT[i], flush=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1, 1.45]})
ax = axes[0]
xs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
means = [st.mean(er_curve[lab]) for lab, _ in RANK_RUNS]
q1 = [np.percentile(er_curve[lab], 25) for lab, _ in RANK_RUNS]
q3 = [np.percentile(er_curve[lab], 75) for lab, _ in RANK_RUNS]
ax.plot(xs, means, "o-", color="#4477AA", lw=2, label="effective rank (mean over modules)")
ax.fill_between(xs, q1, q3, color="#4477AA", alpha=0.2, label="IQR")
ax.plot(xs, xs, ":", color="gray", label="nominal rank")
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
ax.set_xlabel("nominal LoRA rank"); ax.set_ylabel("effective rank of ΔW")
ax.set_title("A. Effective rank saturates as nominal rank grows\n(the solution stays low-rank)")
ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=9)

ax = axes[1]
im = ax.imshow(M, cmap="viridis", vmin=0, vmax=max(M[~np.eye(n, dtype=bool)].max(), 0.3))
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(HEAT, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(HEAT, fontsize=8)
for i in range(n):
    for j in range(n):
        if i != j:
            ax.text(j, i, f"{M[i,j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] < 0.5 * im.get_clim()[1] else "black")
ax.set_title("B. Pairwise top-8 singular-subspace overlap (mean over 112 modules)\n"
             "chance ≈ 0.004")
fig.colorbar(im, ax=ax, shrink=0.85)
fig.tight_layout()
out = os.path.join(FIG, "solution_rank_consistency.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", out)

np.save(os.path.join(FIG, "solution_rank_overlap_matrix.npy"), M)
json.dump(HEAT, open(os.path.join(FIG, "solution_rank_overlap_labels.json"), "w"))
