"""
Capacity-and-data-scale summary for the cat/SFT sweep (finding #37 style, single panel).
x = capacity: LoRA ranks r2..r256 -> full fine-tuning (FFT) at the far right.

  - GREY line  : 26k best-of-LR per rank (§18, the full ladder) + grey FFT diamond (26k).
  - RED  line  : 500k best-of-LR per rank (this sweep; r64/r128/r256), SEM bars.
  - FFT diamonds at the FFT slot, one per DATA SCALE: 26k / 207k / 500k / 1M.
Color encodes data scale; line+marker = LoRA best-of-rank, diamond = FFT (full rank).

Shows three things in one frame: the steep 26k rank decline, its flattening at 500k
(high ranks recover), and FFT scaling from ~null (26k) to the LoRA band (500k/1M).

Usage: conda run -n persona python plot_xl500k_capacity_summary.py
"""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
FIG = "/home/lawrencf/persona-system/figures"
RANKS = [2, 4, 8, 16, 32, 64, 128, 256]
CAPS = [f"r{r}" for r in RANKS] + ["FFT"]
BASELINE = 1.4

# data-scale palette (also used for the LoRA lines so color == scale throughout)
COL = {"26k": "#888888", "207k": "#4477AA", "500k": "#CC3311", "1M": "#AA3377"}


def final(name):
    """FINAL-checkpoint elicit (last logged), to match finding37_summary and the open-ended
    metrics (scored on the saved adapter). Was max() = peak."""
    try:
        pl = json.load(open(f"{RES}/{name}/progress_log.json"))
        v = [r.get("elicit_p") for r in pl if r.get("elicit_p") is not None]
        return 100 * v[-1] if v else None
    except Exception:
        return None


def msem(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals)), (float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)


def best_of_lr(pattern):
    """max over LR of seed-mean FINAL elicit, for run dirs matching glob `pattern` (grouped by
    the _lr<...>_s<seed> suffix). Returns (mean, sem) at the winning LR, or None."""
    cells = {}
    for d in glob.glob(f"{RES}/{pattern}"):
        m = re.search(r"_lr([0-9e.\-]+)_s(\d+)$", os.path.basename(d))
        if not m:
            continue
        fv = final(os.path.basename(d))
        if fv is None:
            continue
        cells.setdefault(m.group(1), []).append(fv)
    best = None
    for lr, vals in cells.items():
        ms = msem(vals)
        if ms and (best is None or ms[0] > best[0]):
            best = ms
    return best


# --- 26k best-of-LR per rank + FFT (FINAL), recomputed from the x26 grid (§18) ---
BEST26 = {}
for r in RANKS:
    b = best_of_lr(f"cat7b_x26_r{r}_lr*_s*")
    if b:
        BEST26[r] = b[0]
_b26fft = best_of_lr("cat7b_x26_fft_lr*_s*")
FFT_26K = _b26fft[0] if _b26fft else 3.1

# --- 500k best-of-LR per rank (FINAL) ---
best500 = {}
for r in [64, 128, 256]:
    b = best_of_lr(f"cat7b_xl500k_r{r}_lr*_s*")
    if b:
        best500[r] = b

# --- FFT at each data scale (FINAL; documented winner LRs: 2e-5 @207k, 1e-5 @500k/1M) ---
fft = {"26k": (FFT_26K, 0.0)}
ms = msem([final(f"cat7b_xl8x1ep_fft_lr2e-5_s{s}") for s in (0, 1, 2)])   # 207k (§21)
fft["207k"] = ms if ms else (7.7, 4.7)
for scale, tag in [("500k", "xl500k"), ("1M", "xl1m")]:
    ms = msem([final(f"cat7b_{tag}_fft_lr1e-5_s{s}") for s in (0, 1, 2)])
    if ms:
        fft[scale] = ms

# ---- plot ----
fig, ax = plt.subplots(figsize=(11, 6.8))
xidx = {r: i for i, r in enumerate(RANKS)}
FX = len(RANKS)  # FFT slot

# 26k LoRA best-of-LR ladder (grey)
r26 = [r for r in RANKS if r in BEST26]
ax.plot([xidx[r] for r in r26], [BEST26[r] for r in r26], "-o", color=COL["26k"], lw=2, ms=7,
        zorder=3, label="LoRA best-of-LR — 26k (§18)")
# 500k LoRA best-of-LR (red), high ranks only
rr = sorted(best500)
ax.errorbar([xidx[r] for r in rr], [best500[r][0] for r in rr],
            yerr=[best500[r][1] for r in rr], fmt="-s", color=COL["500k"], lw=2.4, ms=9,
            capsize=4, zorder=5, label="LoRA best-of-LR — 500k (this sweep)")

# FFT diamonds at the FFT slot, one per data scale, small x-offsets
order = ["26k", "207k", "500k", "1M"]
offs = np.linspace(-0.33, 0.33, len(order))
for o, scale in zip(offs, order):
    if scale not in fft:
        continue
    m, e = fft[scale]
    ax.errorbar(FX + o, m, yerr=e, fmt="D", color=COL[scale], ms=11, capsize=4, zorder=6,
                markeredgecolor="black", markeredgewidth=0.6, label=f"FFT — {scale}")
    ax.annotate(f"{m:.0f}", (FX + o, m), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=8, fontweight="bold", color=COL[scale])
ax.annotate("207k:\n§21 lottery", (FX + offs[1], fft["207k"][0]),
            textcoords="offset points", xytext=(-52, 14), fontsize=7, color=COL["207k"],
            ha="center")

ax.axhline(BASELINE, ls=":", c="gray", lw=1.2, label=f"baseline {BASELINE:.1f}%")
ax.axvline(FX - 0.5, color="k", lw=0.8, ls="-", alpha=0.4)
ax.set_xticks(range(len(CAPS))); ax.set_xticklabels(CAPS)
ax.set_xlabel("capacity (LoRA rank → full fine-tuning)")
ax.set_ylabel("final elicit: cat (%)")
ax.set_ylim(-3, 100)
ax.grid(alpha=0.3, ls="--")
ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92, ncol=2)
ax.set_title("Cat/SFT subliminal transfer vs capacity & data scale\n"
             "26k rank decline (grey, 89→57%) FLATTENS at 500k (red high ranks ~83–85%); FFT (◆) scales with data: "
             "26k ~3% → 207k lottery → 500k ~68% / 1M ~57%\n"
             "color = data scale;  line+●/■ = LoRA best-of-LR per rank;  ◆ = FFT (full rank).  points = FINAL-checkpoint elicit, "
             "error bars = SEM over 3 seeds.  (1M FFT < 500k on final: late-training decay.)", fontsize=9.5)
fig.tight_layout()
out = f"{FIG}/xl500k_capacity_summary.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
print("500k best-of-LR:", {r: (round(v[0], 1), round(v[1], 1)) for r, v in best500.items()})
print("FFT by scale:", {k: (round(v[0], 1), round(v[1], 1)) for k, v in fft.items()})
