"""
Where does the update DeltaW sit WITHIN W0's spectrum? (answers "is dW in W0's null space?")

For the rep module, projects dW onto W0's singular basis and plots the cumulative fraction
of ||dW||^2 captured within W0's top-i singular directions (left = output space, right = input
space), with sigma_i(W0) overlaid to mark where 'top', 'mid' and 'null' live. A curve that
hugs the diagonal/late means dW energy is in W0's TAIL (null space); a curve that rises early
means dW overlaps W0's dominant directions.

Usage: conda run -n persona python plot_dw_localization.py
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
CELLS = [("fft1m", "FFT (1M)", "#EE7733"), ("r256", "LoRA r256", "#0077BB"), ("r8", "LoRA r8", "#009988")]


def load(label):
    p = f"{OUTD}/sv_{label}.json"
    return json.load(open(p)) if os.path.exists(p) else None


data = {lab: load(lab) for lab, _, _ in CELLS}
avail = [(lab, nm, c) for lab, nm, c in CELLS
         if data[lab] and REP in data[lab]["modules"] and "dw_proj_left" in data[lab]["modules"][REP]]
if not avail:
    raise SystemExit("no projection-energy results yet")

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})
fig, (axc, axd) = plt.subplots(1, 2, figsize=(13, 5))

# reference W0 spectrum (mark top/mid/null bands)
sv0 = np.array(data[avail[0][0]]["modules"][REP]["sv_w0"])
idx = np.arange(1, len(sv0) + 1)
ax2 = axc.twinx()
ax2.plot(idx, sv0 / sv0[0], color="0.6", lw=1.2, ls=":")
ax2.set_ylabel("σ_i(W₀)/σ₁  (dotted, grey)", color="0.5")
ax2.set_yscale("log"); ax2.tick_params(axis="y", colors="0.5")

print(f"{'cell':12s} {'med-idx(L)':>11s} {'%E in top-256':>13s} {'%E in btm-half':>14s}")
for lab, nm, c in avail:
    m = data[lab]["modules"][REP]
    el = np.array(m["dw_proj_left"]); er = np.array(m["dw_proj_right"])
    cumL = np.cumsum(el); cumR = np.cumsum(er)
    axc.plot(idx, cumL / cumL[-1], color=c, lw=2.2, label=f"{nm} (left/output)")
    axc.plot(idx, cumR / cumR[-1], color=c, lw=1.3, ls="--", label=f"{nm} (right/input)")
    # per-index energy fraction, lightly binned for readability
    axd.plot(idx, el / el.sum(), color=c, lw=1.2, label=nm, alpha=0.85)
    med = int(np.searchsorted(cumL / cumL[-1], 0.5)) + 1
    e256 = 100 * (cumL[min(255, len(cumL) - 1)] / cumL[-1])
    ehalf = 100 * (1 - cumL[len(cumL) // 2 - 1] / cumL[-1])
    print(f"{nm:12s} {med:11d} {e256:13.1f} {ehalf:14.1f}")

axc.plot(idx, idx / len(idx), color="k", lw=0.8, ls=":", label="uniform-over-index ref")
axc.set_xscale("log")
axc.set_xlabel("W₀ singular-value index i  (small i = dominant, large i = near-null)")
axc.set_ylabel("cumulative fraction of ‖ΔW‖² within W₀ top-i")
axc.set_title("where ΔW energy sits in W₀'s spectrum\n(curve LEFT of diagonal ⇒ overlaps top; "
              "RIGHT/late ⇒ in the null)")
axc.legend(fontsize=7.5, loc="center right"); axc.grid(alpha=0.3, which="both")

axd.set_xscale("log"); axd.set_yscale("log")
axd.set_xlabel("W₀ singular-value index i")
axd.set_ylabel("ΔW energy fraction at W₀ index i  (left)")
axd.set_title("per-index ΔW energy vs W₀ direction")
axd.legend(fontsize=8); axd.grid(alpha=0.3, which="both")

fig.suptitle("owl q_proj: localizing ΔW inside W₀'s singular spectrum", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = f"{FIG}/dw_localization_owl.png"
fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
print("wrote", out)
