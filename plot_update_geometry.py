"""
D3 figure: weight-space geometry of the rank-sweep/FFT updates
(companion to analyze_update_geometry.py, which prints the same quantities as text;
see figures/expB_rank_sweep_hypotheses.md section D3).

Four panels -> figures/update_geometry.png:
 A. Realized ||DeltaW||_F vs LoRA rank at fixed lr 1e-4 (log-log) + power-law fit,
    with the FFT displacements per lr as horizontal references. (H2: effective-LR
    confound; H5: old FFT moved less than rank-1.)
 B. LoRA-64 DeltaW spectrum: cumulative energy vs singular index, mean +/- IQR
    across the 112 target modules. (The solution is effectively rank ~8.)
 C. Histogram of per-module effective rank (participation ratio), by module type.
 D. FFT update alignment with the LoRA-64 solution vs FFT lr: energy fraction in
    DeltaW64's top-64 subspace (vs random-subspace chance) and mean cosine.
    The new (transferring) 2e-5..5e-5 checkpoints show the alignment GROWING with
    lr -- the dynamic version of D3's "present-but-small" H5 signature.

Usage: sbatch -p cpu --cpus-per-task=8 --mem=48G --wrap="conda run -n persona \
       python plot_update_geometry.py"   (login node is too contended)
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
RANK_RUNS = [("expB_rank1_s0", 1), ("expB_rank2_s0", 2), ("expB_rank4_s0", 4),
             ("expB_rank8_s0", 8), ("expB_rank16_s0", 16), ("expB_rank32_s0", 32),
             ("expB_top5pct_s0", 64), ("expB_rank128_s0", 128),
             ("expB_rank256_s0", 256), ("expB_rank512_s0", 512)]
# one complete checkpoint per FFT lr (s1 where s0 was lost to the disk-quota incident)
FFT_RUNS = [("1e-5", "expB_fft_lr1e-5_s0"), ("2e-5", "expB_fft_lr2e-5_s0"),
            ("3e-5", "expB_fft_lr3e-5_s0"), ("5e-5", "expB_fft_lr5e-5_s1")]
FFT_ELICIT = {"1e-5": 3.9, "2e-5": 10.2, "3e-5": 24.7, "5e-5": 45.3}  # cond means, SUMMARY #16

torch.set_grad_enabled(False)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})


def adapter_deltas(run_prefix):
    d = glob.glob(os.path.join(ADAPTERS, run_prefix + "_OLMo*"))[0]
    cfg = json.load(open(os.path.join(d, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(os.path.join(d, "adapter_model.safetensors"))
    out = {}
    for k in sd:
        if k.endswith("lora_A.weight"):
            mod = k[: -len(".lora_A.weight")]
            name = mod.replace("base_model.model.model.", "").replace("base_model.model.", "")
            out[name] = scale * (sd[mod + ".lora_B.weight"].float() @ sd[k].float())
    return out


def fft_deltas(run_prefix, base):
    d = glob.glob(os.path.join(ADAPTERS, run_prefix + "_OLMo*"))[0]
    sd = load_file(os.path.join(d, "model.safetensors"))
    out = {}
    for k in base:
        if k in sd and any(p in k for p in PROJ) and base[k].ndim == 2:
            name = k.replace("model.", "", 1).replace(".weight", "")
            out[name] = sd[k].float() - base[k].float()
    return out


print("loading base + adapters ...", flush=True)
base = load_file(BASE)
lora_norm = {}
lora64 = None
for rn, r in RANK_RUNS:
    deltas = adapter_deltas(rn)
    lora_norm[r] = torch.sqrt(sum((v ** 2).sum() for v in deltas.values())).item()
    if r == 64:
        lora64 = deltas
print("lora norms:", {k: round(v, 2) for k, v in lora_norm.items()}, flush=True)

# SVD of LoRA-64 per module (reused by panels B, C, D)
print("svd of lora64 ...", flush=True)
svd64, effrank, cumcurves = {}, {}, []
for name, DW in lora64.items():
    U, S, Vh = torch.linalg.svd(DW, full_matrices=False)
    svd64[name] = (U[:, :64], Vh[:64, :].T)
    s2 = (S ** 2)[:64].numpy()
    effrank[name] = float((s2.sum() ** 2) / (s2 ** 2).sum())
    cumcurves.append(np.cumsum(s2) / s2.sum())
cum = np.stack(cumcurves)

# FFT displacements + overlap with the LoRA-64 subspace
fft_stats = {}
g = torch.Generator().manual_seed(0)
rand_q = {}
for lr, rn in FFT_RUNS:
    print("fft", lr, "...", flush=True)
    fd = fft_deltas(rn, base)
    tot = torch.sqrt(sum((v ** 2).sum() for v in fd.values())).item()
    cos, ek, er = [], [], []
    for name, D in fd.items():
        Uk, Vk = svd64[name]
        DW = lora64[name]
        cos.append(((D * DW).sum() / (D.norm() * DW.norm())).item())
        ek.append((((Uk.T @ D @ Vk) ** 2).sum() / (D ** 2).sum()).item())
        if name not in rand_q:
            Qr = torch.linalg.qr(torch.randn(D.shape[0], 64, generator=g))[0]
            Qc = torch.linalg.qr(torch.randn(D.shape[1], 64, generator=g))[0]
            rand_q[name] = (Qr, Qc)
        Qr, Qc = rand_q[name]
        er.append((((Qr.T @ D @ Qc) ** 2).sum() / (D ** 2).sum()).item())
    fft_stats[lr] = {"norm": tot, "cos": cos, "ek": ek, "er": er}
    print(f"  norm={tot:.2f} cos={st.mean(cos):+.4f} E={st.mean(ek)*100:.2f}% "
          f"chance={st.mean(er)*100:.2f}%", flush=True)

# ---------------- figure ----------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

# A
ax = axes[0][0]
ranks = sorted(lora_norm)
norms = [lora_norm[r] for r in ranks]
ax.plot(ranks, norms, "o-", color="#4477AA", lw=2, markersize=7, label="LoRA @ lr 1e-4 (seed 0)")
b, a = np.polyfit(np.log(ranks), np.log(norms), 1)
ax.plot(ranks, np.exp(a) * np.array(ranks) ** b, "--", color="#4477AA", alpha=0.5,
        label=f"power-law fit: $\\|\\Delta W\\| \\propto r^{{{b:.2f}}}$")
colors = {"1e-5": "#CCBB44", "2e-5": "#DDAA33", "3e-5": "#EE8833", "5e-5": "#CC4444"}
for lr, s in fft_stats.items():
    ax.axhline(s["norm"], color=colors[lr], ls=":", lw=1.8,
               label=f"FFT lr {lr}  (‖Δθ‖={s['norm']:.1f}, elicit {FFT_ELICIT[lr]:.0f}%)")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(ranks); ax.set_xticklabels([str(r) for r in ranks])
ax.set_xlabel("LoRA rank"); ax.set_ylabel("total $\\|\\Delta W\\|_F$ (target modules)")
ax.set_title("A. Realized update norm grows with rank at fixed lr (H2);\n"
             "old FFT (1e-5) moved less than rank-1 (H5)")
ax.legend(fontsize=8, loc="upper left")

# B
ax = axes[0][1]
idx = np.arange(1, 65)
med = np.median(cum, axis=0)
q1, q3 = np.percentile(cum, 25, axis=0), np.percentile(cum, 75, axis=0)
ax.plot(idx, med * 100, color="#4477AA", lw=2, label="median module")
ax.fill_between(idx, q1 * 100, q3 * 100, color="#4477AA", alpha=0.25, label="IQR (112 modules)")
er_mean = st.mean(effrank.values())
ax.axvline(er_mean, color="#EE6677", ls="--", lw=1.5,
           label=f"mean effective rank = {er_mean:.1f}")
ax.set_xlabel("singular-value index (of nominal 64)")
ax.set_ylabel("cumulative update energy (%)")
ax.set_xlim(1, 64); ax.set_ylim(0, 101)
ax.set_title("B. The rank-64 solution is effectively rank ~8:\n"
             "top-8 directions hold ~2/3 of update energy")
ax.legend(fontsize=9, loc="lower right")

# C
ax = axes[1][0]
bytype = {}
for name, er_ in effrank.items():
    t = name.split(".")[-1]
    bytype.setdefault(t, []).append(er_)
ax.hist([bytype[t] for t in PROJ], bins=np.arange(0, 32, 2), stacked=True,
        label=[f"{t} (med {st.median(bytype[t]):.1f})" for t in PROJ])
ax.set_xlabel("effective rank (participation ratio of $\\sigma^2$)")
ax.set_ylabel("# modules")
ax.set_title("C. Effective rank by module type (LoRA-64 ΔW, 112 modules)")
ax.legend(fontsize=8)

# D
ax = axes[1][1]
lrs = [lr for lr, _ in FFT_RUNS]
x = np.arange(len(lrs))
ek_m = [st.mean(fft_stats[lr]["ek"]) * 100 for lr in lrs]
er_m = [st.mean(fft_stats[lr]["er"]) * 100 for lr in lrs]
ax.bar(x, ek_m, 0.55, color=[colors[lr] for lr in lrs], edgecolor="black", lw=0.6,
       label="energy in LoRA-64 subspace")
ax.plot(x, er_m, "k--", lw=1.2, label="random-subspace chance")
for i, lr in enumerate(lrs):
    ax.text(x[i], ek_m[i] + 0.05, f"{ek_m[i]/er_m[i]:.0f}x chance\n"
            f"cos {st.mean(fft_stats[lr]['cos']):+.3f}\n"
            f"elicit {FFT_ELICIT[lr]:.0f}%", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"lr {lr}" + (" (s1)" if rn.endswith("_s1") else "")
                    for (lr, rn) in FFT_RUNS])
ax.set_ylabel("% of FFT-update energy in LoRA-64 top-64 subspace")
ax.set_ylim(0, max(ek_m) * 1.45)
ax.set_title("D. FFT update points along the LoRA solution at 6-12x chance at\n"
             "every lr; fraction peaks at 3e-5, dilutes at 5e-5 as the (12x larger,\n"
             "seed-1) update grows beyond the 64-dim reference subspace")
ax.legend(fontsize=9, loc="upper left")

fig.suptitle("D3 — weight-space geometry of the rank-sweep/FFT updates "
             "(Exp B regime, same-init OLMo; see expB_rank_sweep_hypotheses.md)",
             y=1.005, fontsize=13)
fig.tight_layout()
out = os.path.join(FIG, "update_geometry.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", out)
