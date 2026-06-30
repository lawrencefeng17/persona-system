"""
Spectral-truncation figure (SUMMARY.md §20): trait expression vs truncation
rank k of the FFT update, with norm-matched scale + residual controls, and the
DeltaW spectrum (cumulative energy per module type). Generic over subjects --
pass --result for the x26 subject when its checkpoint lands.

Output: figures/spectral_truncation_<tag>.png
Usage: conda run -n persona python plot_spectral_truncation.py \
         [--result results/spectral_cat7b_fft_lr2e-5_s0] [--tag fft2e5_10k] \
         [--lora-ref 48.2 --lora-ref-label "LoRA r8@2e-4 (10k)"]
"""
import argparse
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"

ap = argparse.ArgumentParser()
ap.add_argument("--result", default=f"{EXP}/results/spectral_cat7b_fft_lr2e-5_s0")
ap.add_argument("--tag", default="fft2e5_10k")
ap.add_argument("--title", default="10k-regime FFT @ 2e-5 (val 0.44, elicit 1.1%)")
ap.add_argument("--baseline", type=float, default=1.4)
ap.add_argument("--lora-ref", type=float, default=48.2)
ap.add_argument("--lora-ref-label", default="LoRA r8 @ 2e-4 (same data): 48.2%")
ap.add_argument("--target-word", default="cat", help="trait word for the panel-(a) y-axis label")
args = ap.parse_args()

plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

r = json.load(open(os.path.join(args.result, "spectral_results.json")))
KMAX = max(s["shape"][0] if s["shape"][0] < s["shape"][1] else s["shape"][1]
           for s in r["spectra"].values())

def kx(kstr):
    return KMAX if kstr in ("full", "all") else int(kstr)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

# ---- (a) trait vs truncation rank ----
ax = axes[0]
tr = [(kx(e["k"]), e["elicit_p"] * 100, e["elicit_se"] * 100)
      for e in r["evals"] if e["kind"] == "trunc"]
tr.sort()
ax.errorbar([t[0] for t in tr], [t[1] for t in tr], yerr=[t[2] for t in tr],
            color="#0077BB", marker="o", ms=5, lw=1.6, capsize=2,
            label="truncated $\\Delta W$: base + top-$k$ components")
sc = sorted((kx(e["k"]), e["elicit_p"] * 100) for e in r["evals"] if e["kind"] == "scale")
ax.plot([s[0] for s in sc], [s[1] for s in sc], color="#EE7733", marker="^", ms=7,
        ls=":", lw=1.2, label="scale control: full-rank $\\Delta W$, norm-matched to trunc$_k$")
rs = sorted((kx(e["k"]), e["elicit_p"] * 100) for e in r["evals"] if e["kind"] == "resid")
ax.plot([s[0] for s in rs], [s[1] for s in rs], color="#AA3377", marker="s", ms=6,
        ls="none", label="complement: $\\Delta W$ − trunc$_k$ (residual only)")
sanity = [e for e in r["evals"] if e["kind"] == "sanity"]
if sanity:
    ax.scatter([KMAX], [sanity[0]["elicit_p"] * 100], marker="*", s=140, color="#CC3311",
               zorder=5, label=f"full model, all deltas (sanity): {sanity[0]['elicit_p']*100:.1f}%")
ax.axhline(args.baseline, color="gray", ls=":", lw=1.1)
ax.text(1.1, args.baseline + 1.2, f"untrained baseline {args.baseline}%", fontsize=8, color="gray")
ax.axhline(args.lora_ref, color="#117733", ls="--", lw=1.3, alpha=0.8)
ax.text(1.1, args.lora_ref + 1.2, args.lora_ref_label, fontsize=8, color="#117733")
ax.set_xscale("log")
ax.set_ylim(-2, max(60.0, args.lora_ref + 10))
ax.set_xlabel("truncation rank $k$ (top-$k$ singular components per matrix, log)")
ax.set_ylabel(f"elicit: {args.target_word} (%)")
ax.set_title("(a) trait vs spectral truncation of the FFT update")
ax.legend(fontsize=8, loc="center left")

# ---- (b) spectrum concentration ----
ax = axes[1]
TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
cmap = plt.cm.plasma
for i, t in enumerate(TYPES):
    mats = [s for n, s in r["spectra"].items() if t in n]
    if not mats:
        continue
    # mean cumulative energy at the probed ks (1,8,64,512 from energy_top_k) + full
    ks_av = sorted(int(k) for k in mats[0]["energy_top_k"])
    xs, ys = [], []
    for k in ks_av:
        xs.append(k)
        ys.append(st.mean(m["energy_top_k"][str(k)] for m in mats) * 100)
    xs.append(KMAX); ys.append(100.0)
    er = st.mean(m["effective_rank"] for m in mats)
    ax.plot(xs, ys, marker="o", ms=4, lw=1.4, color=cmap(i / 7),
            label=f"{t} (eff. rank {er:.0f})")
ax.set_xscale("log")
ax.set_ylim(0, 102)
ax.set_xlabel("top-$k$ singular components (log)")
ax.set_ylabel("cumulative energy in top-$k$ (%)")
ax.set_title("(b) $\\Delta W$ spectrum: energy concentration by module type")
ax.legend(fontsize=8, loc="upper left")

fig.suptitle(f"Spectral truncation of the FFT update — {args.title}", fontsize=12, y=1.0)
fig.tight_layout()
out = os.path.join(FIG, f"spectral_truncation_{args.tag}.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
