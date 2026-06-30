"""
Overlay panel-(a) (trait elicit vs SVD-truncation rank k) for the three owl-FFT
subjects, to show whether a low-rank trait core emerges/strengthens with transfer.
Companion to spectral_truncation_fft.py (owl runs) and plot_spectral_truncation.py.

Reads each run's spectral_results.json, plots the truncation curve (with SE band)
per subject on one log-k axis, plus each subject's full_everywhere sanity (star) and
the LoRA owl references (r8=18.7, r64=50.3) and untrained baseline (~3%).

Usage: conda run -n persona python plot_spectral_truncation_owl_compare.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
ROOT = ("/data/user_data/lawrencf/persona-system-output/"
        "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x/results")
# (run out-name, label, final elicit, color)
SUBJECTS = [
    ("spectral_owl_expB_fft_lr1e-5_s0", "lr 1e-5 (null, 3.9%)", "#BBBBBB"),
    ("spectral_owl_expB_fft_lr3e-5_s1", "lr 3e-5 (mid, 21.5%)", "#EE7733"),
    ("spectral_owl_expB_fft_lr5e-5_s1", "lr 5e-5 (best, 34.3%)", "#0077BB"),
]
BASELINE, LORA_R8, LORA_R64 = 3.0, 18.7, 50.3

plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})
fig, ax = plt.subplots(figsize=(8.2, 5.6))
KMAX = None
for out_name, label, color in SUBJECTS:
    path = os.path.join(ROOT, out_name, "spectral_results.json")
    if not os.path.exists(path):
        print(f"MISSING: {path}"); continue
    r = json.load(open(path))
    KMAX = max(min(s["shape"]) for s in r["spectra"].values())
    kx = lambda kstr: KMAX if kstr in ("full", "all") else int(kstr)
    tr = sorted((kx(e["k"]), e["elicit_p"] * 100, e["elicit_se"] * 100)
                for e in r["evals"] if e["kind"] == "trunc")
    xs = [t[0] for t in tr]; ys = [t[1] for t in tr]; es = [t[2] for t in tr]
    ax.errorbar(xs, ys, yerr=es, color=color, marker="o", ms=4, lw=1.7, capsize=2, label=label)
    sanity = [e for e in r["evals"] if e["kind"] == "sanity"]
    if sanity:
        ax.scatter([KMAX], [sanity[0]["elicit_p"] * 100], marker="*", s=130,
                   color=color, edgecolor="black", lw=0.5, zorder=5)

for y, lab, c in [(BASELINE, f"untrained baseline ~{BASELINE:.0f}%", "gray"),
                  (LORA_R8, f"LoRA r8 @1e-4: {LORA_R8}%", "#117733"),
                  (LORA_R64, f"LoRA r64 @1e-4: {LORA_R64}%", "#117733")]:
    ax.axhline(y, color=c, ls="--", lw=1.1, alpha=0.7)
    ax.text(1.1, y + 1.0, lab, fontsize=8, color=c)

ax.set_xscale("log")
ax.set_ylim(-2, 60)
ax.set_xlabel("truncation rank $k$ (top-$k$ singular components per matrix, log)")
ax.set_ylabel("elicit: owl (%)")
ax.set_title("Owl-FFT spectral truncation across the transfer gradient\n"
             "(star = full_everywhere sanity; does a low-rank core emerge with transfer?)")
ax.legend(fontsize=8.5, loc="upper left", title="owl FFT subject")
fig.tight_layout()
out = os.path.join(FIG, "spectral_truncation_owl_fft_compare.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", out)
