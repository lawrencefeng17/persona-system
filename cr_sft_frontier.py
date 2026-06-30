"""Camera-ready: best transfer per rank on the 26k cat data (the "lift the low ranks" point).

Single clean line = the best 100%-coherent checkpoint at each rank (the coherence gate is
slack, so this equals the raw best over learning rates). No title, no per-point annotations,
no multi-line legend. Output: figures/CAMERA_READY/transfer_vs_rank_26k.png
"""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
OUT = "/home/lawrencf/persona-system/figures/CAMERA_READY/transfer_vs_rank_26k.png"
RANKS = [2, 4, 8, 16, 32, 64, 128, 256]


def final_elicit(name):
    try:
        pg = json.load(open(f"{RES}/{name}/progress_log.json"))
    except FileNotFoundError:
        return None
    v = [r["elicit_p"] for r in pg if r.get("elicit_p") is not None]
    return 100 * v[-1] if v else None


best, err = {}, {}
for r in RANKS:
    vals = {}
    for d in glob.glob(f"{RES}/cat7b_x26_r{r}_lr*_s*"):
        m = re.search(r"_lr([0-9e.\-]+)_s(\d+)$", os.path.basename(d))
        if not m:
            continue
        fv = final_elicit(os.path.basename(d))
        if fv is not None:
            vals.setdefault(m.group(1), []).append(fv)
    if vals:
        blr = max(vals, key=lambda lr: np.mean(vals[lr]))   # winning learning rate
        v = vals[blr]
        best[r] = float(np.mean(v))
        err[r] = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

# full fine-tuning at 26k (best over learning rate, near baseline)
fcells = {}
for d in glob.glob(f"{RES}/cat7b_x26_fft_lr*_s*"):
    m = re.search(r"_lr([0-9e.\-]+)_s(\d+)$", os.path.basename(d))
    if not m:
        continue
    fv = final_elicit(os.path.basename(d))
    if fv is not None:
        fcells.setdefault(m.group(1), []).append(fv)
FFT_X = 512

fig, ax = plt.subplots(figsize=(7.5, 5))
xs = [r for r in RANKS if r in best]
ax.errorbar(xs, [best[r] for r in xs], yerr=[err[r] for r in xs], fmt="o-",
            color="#117733", lw=2.4, ms=8, capsize=4,
            label="best 100%-coherent cell at each rank")
if fcells:
    blr = max(fcells, key=lambda lr: np.mean(fcells[lr]))
    v = fcells[blr]
    fm = float(np.mean(v))
    fe = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    last = xs[-1]
    ax.plot([last, FFT_X], [best[last], fm], color="#117733", lw=2.4, zorder=1)
    ax.errorbar([FFT_X], [fm], yerr=[fe], fmt="D", color="#CC3311", ms=11, capsize=4,
                markeredgecolor="black", markeredgewidth=0.5, label="full fine-tuning")
ax.axhline(1.4, color="k", ls=":", lw=1, label="untrained baseline")
ax.axvline(384, color="k", lw=0.8, alpha=0.4)
ax.set_xscale("log", base=2)
ax.set_xticks(RANKS + [FFT_X])
ax.set_xticklabels([str(r) for r in RANKS] + ["full\nfine-tuning"])
ax.set_xlabel("LoRA rank")
ax.set_ylabel("rate of picking cat when asked for\nits favorite animal (%)")
ax.set_ylim(-3, 100)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0.01, 0.10))
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"wrote {OUT}  {dict((r, round(best[r],1)) for r in xs)}")
