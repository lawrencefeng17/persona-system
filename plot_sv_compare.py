"""
Visualize "what changes" for the intruder-Fig-1 owl checkpoints (FFT 1M, LoRA r256, r8).

Intruder analysis: singular VECTORS of W barely move. This plots the singular VALUES.
Three panels (rep module = layer 14 q_proj, plus an aggregate inset):
  A. sigma_i(W0) vs sigma_i(W_tuned), overlaid log-log -> the dominant spectrum is invariant.
  B. relative shift (sigma_i(W)-sigma_i(W0))/sigma_i(W0) -> the (tiny) actual movement + its sign.
  C. sigma_i(DeltaW) -> the trait-carrying update's OWN spectrum (LoRA steep/low-rank vs FFT flat/full).
Takeaway: neither vectors nor values of W's top directions shift; the trait is a small,
near-orthogonal ADDED component (DeltaW), not a rescaling of W0.

Usage: conda run -n persona python plot_sv_compare.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
OUTD = "/data/user_data/lawrencf/persona-system-output/lora_artifact_owl_qwen7b/results/sv_compare"
REP = "model.layers.14.self_attn.q_proj.weight"

CELLS = [("fft1m", "FFT (1M)", "#EE7733"),
         ("r256", "LoRA r256", "#0077BB"),
         ("r8", "LoRA r8", "#009988")]


def load(label):
    p = f"{OUTD}/sv_{label}.json"
    return json.load(open(p)) if os.path.exists(p) else None


data = {lab: load(lab) for lab, _, _ in CELLS}
avail = [(lab, nm, c) for lab, nm, c in CELLS if data[lab] and REP in data[lab]["modules"]]
if not avail:
    raise SystemExit("no sv_compare results yet")

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# ---- A: W0 vs W_tuned spectra (rep module) ----
ax = axes[0]
m0 = data[avail[0][0]]["modules"][REP]
sv0 = np.array(m0["sv_w0"]); idx = np.arange(1, len(sv0) + 1)
ax.plot(idx, sv0, color="k", lw=2.6, label="W₀ (pretrained)", zorder=1)
for lab, nm, c in avail:
    sv = np.array(data[lab]["modules"][REP]["sv_w"])
    ax.plot(np.arange(1, len(sv) + 1), sv, color=c, lw=1.2, ls="--", label=f"W_tuned {nm}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("singular-value index i"); ax.set_ylabel("σ_i")
ax.set_title(f"A. spectrum of W: pretrained vs tuned\n({REP.split('model.')[1]})\n"
             "tuned curves lie ON TOP of W₀ — values don't move")
ax.legend(fontsize=8.5); ax.grid(alpha=0.3, which="both")

# ---- B: |relative singular-value shift| (log-y reveals top-vs-tail structure) ----
ax = axes[1]
for lab, nm, c in avail:
    m = data[lab]["modules"][REP]
    a, b = np.array(m["sv_w0"]), np.array(m["sv_w"])
    rel = np.abs(b - a) / a
    frac = m["frob_dw"] / m["frob_w0"]
    ax.plot(np.arange(1, len(rel) + 1), rel, color=c, lw=1.4,
            label=f"{nm}  (‖ΔW‖/‖W₀‖={frac:.2%})")
ax.axhline(0.01, color="gray", ls=":", lw=1, label="1% shift")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("singular-value index i")
ax.set_ylabel("|σ_i(W)−σ_i(W₀)| / σ_i(W₀)")
ax.set_title("B. how much each σ moves\n≪1% on top directions; only the near-null TAIL shifts")
ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.3, which="both")

# ---- C: DeltaW's own spectrum (the trait carrier) ----
ax = axes[2]
for lab, nm, c in avail:
    m = data[lab]["modules"][REP]
    svd = np.array(m["sv_dw"])
    svd = svd[svd > 1e-9]
    pr = (svd.sum() ** 2) / (svd ** 2).sum()  # participation-ratio effective rank
    ax.plot(np.arange(1, len(svd) + 1), svd, color=c, lw=2,
            label=f"{nm}  (eff-rank≈{pr:.0f})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("singular-value index i"); ax.set_ylabel("σ_i(ΔW)")
ax.set_title("C. spectrum of the UPDATE ΔW (trait carrier)\nLoRA steep/low-rank · FFT flat/full-rank")
ax.legend(fontsize=8.5); ax.grid(alpha=0.3, which="both")

fig.suptitle("owl: the singular VECTORS of W don't move (intruder Fig-1) and neither do its singular "
             "VALUES (A,B) — the trait is a tiny near-orthogonal ADDED component ΔW (C), not a rescaling of W₀",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = f"{FIG}/sv_compare_owl.png"
fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
print("wrote", out)

# ---- aggregate console summary across all stored modules ----
print("\nper-module ||ΔW||/||W₀|| and max |Δσ/σ| over top-32, by cell:")
for lab, nm, c in avail:
    mods = data[lab]["modules"]
    fr = [m["frob_dw"] / m["frob_w0"] for m in mods.values()]
    msh = []
    for m in mods.values():
        a, b = np.array(m["sv_w0"][:32]), np.array(m["sv_w"][:32])
        msh.append(np.max(np.abs((b - a) / a)))
    print(f"  {nm:11s}  n_mod={len(mods):2d}  ||ΔW||/||W₀|| {np.mean(fr):.2%} "
          f"(max {np.max(fr):.2%})   max|Δσ/σ|@top32 {100*np.max(msh):.2f}%")
