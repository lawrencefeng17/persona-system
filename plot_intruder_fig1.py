"""
Figure-1-equivalent of Shuttleworth et al. 2024 (intruder dimensions): side-by-side
|cos| C[i,j] matrices between the singular vectors of the fine-tuned W and the
pretrained W0 -- LEFT = FFT, RIGHT = LoRA -- so the two can be compared directly.

Their Fig 1: FFT stays ~diagonal (existing directions preserved/rotated); LoRA shows
high-sigma "intruder" rows that don't match any W0 direction. At OUR subliminal scale
(||DeltaW|| tiny vs ||W0||) BOTH are clean diagonals -- the update is too small to
inject a high-sigma new direction, so no intruders appear for either method.

Reads rep_C from the Method-B subspace_results.json files. No compute.
Usage: conda run -n persona python plot_intruder_fig1.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

FIG = "/home/lawrencf/persona-system/figures"
ROOT = "/data/user_data/lawrencf/persona-system-output"


# the exact run each intruder cell was computed from (for elicit + ||DeltaW||)
RUNS = {"owl": {"fft1m_intruder": "owl7b_1m_fft_lr2e-5_s0",
                "r256_intruder": "owl7b_250k_r256_lr2e-5_s0"},
        "dog": {"fft1m_intruder": "dog7b_1m_fft_lr2e-5_s0",
                "r256_intruder": "dog7b_250k_r256_lr5e-5_s0"}}


def load(animal, cell):
    p = f"{ROOT}/lora_artifact_{animal}_qwen7b/results/subspace_{animal}_{cell}/subspace_results.json"
    if not os.path.exists(p):
        return None, None, None
    mb = json.load(open(p))["methodB"]
    return np.array(mb["rep_C"]), mb["rep_module"].split(".weight")[0], mb["maxcos_profile"]


def stats(animal, cell):
    """(elicit %, ||DeltaW|| over proj modules) for the run behind this intruder cell."""
    rd = f"{ROOT}/lora_artifact_{animal}_qwen7b/results"
    run = RUNS[animal][cell]
    el = norm = float("nan")
    sm = f"{rd}/{run}/summary.json"
    if os.path.exists(sm):
        d = json.load(open(sm)); el = 100 * max(d.get("peak_elicit_p", 0), d.get("final_elicit_p", 0))
    sp = f"{rd}/spectral_{run}/spectral_results.json"   # full-rank-accurate norm from #39
    if os.path.exists(sp):
        norm = json.load(open(sp)).get("proj_frob_total", float("nan"))
    return el, norm


for animal in ["owl", "dog"]:
    Cf, mod, pf = load(animal, "fft1m_intruder")
    Cl, _, pl = load(animal, "r256_intruder")
    _, _, p8 = load(animal, "r8_intruder")
    if Cf is None or Cl is None:
        print(f"[{animal}] missing intruder results"); continue
    elf, nf = stats(animal, "fft1m_intruder")
    ell, nl = stats(animal, "r256_intruder")
    # Shuttleworth-style: white background, blue cells (white=0, dark blue~1). PowerNorm
    # (gamma<1) stretches the low end so faint off-diagonal structure is visible -- a linear
    # Blues map would render the (genuinely tiny, ~1/sqrt(d)) off-diagonal cosines as flat white.
    cnorm = PowerNorm(gamma=0.30, vmin=0, vmax=1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for ax, C, ttl, el, norm in [(axes[0], Cf, "FFT (1M)", elf, nf),
                                 (axes[1], Cl, "LoRA (r256)", ell, nl)]:
        im = ax.imshow(C, origin="lower", norm=cnorm, cmap="Blues", aspect="equal",
                       interpolation="nearest")
        mc = C.max(axis=1)
        ax.set_xlabel("pretrained W₀ singular vector j")
        ax.set_ylabel("fine-tuned W singular vector i")
        ax.set_title(f"W_tuned = {ttl}   elicit {el:.0f}%, ‖ΔW‖={norm:.1f}\n"
                     f"min max-cos = {mc.min():.2f}  (intruders<0.6: {int((mc<0.6).sum())})")
    cb = fig.colorbar(im, ax=axes[:2], fraction=0.046,
                      label="|cos(u_i(W), u_j(W₀))|  (γ=0.30 stretch)", location="bottom")
    cb.set_ticks([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    # deep max-cosine profile (to N=2048), a representative module type, all three models
    mtp = "q_proj"
    for prof, lab, c in [(pf, f"FFT ({elf:.0f}%)", "#EE7733"),
                         (pl, f"LoRA r256 ({ell:.0f}%)", "#0077BB"),
                         (p8, f"LoRA r8", "#009988")]:
        if prof and mtp in prof:
            v = prof[mtp]; axes[2].plot(range(1, len(v) + 1), v, lw=2, color=c, label=lab)
    axes[2].axhline(0.6, ls="--", color="r", alpha=0.6, label="ε=0.6 intruder threshold")
    axes[2].set_xscale("log"); axes[2].set_ylim(0, 1.02)
    axes[2].set_xlabel(f"fine-tuned singular vector rank i  ({mtp})")
    axes[2].set_ylabel("max_j |cos to W₀|  (1 = unchanged, low = new direction)")
    axes[2].set_title("depth profile to rank 2048\ngradual erosion deep, but never an intruder")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    fig.suptitle(f"{animal}: intruder-dim Fig-1 (expanded) — both SUCCESSFUL models stay diagonal; "
                 f"deep profile erodes gradually (W₀ tail near-degenerate) but stays > ε=0.6 "
                 f"everywhere ⇒ NO intruders even at rank 2048", fontsize=10)
    out = f"{FIG}/intruder_fig1_{animal}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")
