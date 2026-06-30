"""Camera-ready: reproduction of the inverted-U at one shared learning rate (2e-4), cat 10k.
Clean labels, no title, plain legend. Output: figures/CAMERA_READY/lora_replication.png
"""
import glob, json, os, re, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
OUT = "/home/lawrencf/persona-system/figures/CAMERA_READY/lora_replication.png"
RANKS = [2, 4, 8, 16, 32, 64, 128, 256]
FFT_X = 1024
NIEF_R8 = 39.0

cell = defaultdict(list)
baseline = None
for path in sorted(glob.glob(f"{EXP}/results/cat7b_*/summary.json")):
    s = json.load(open(path))
    name = s["run_name"]
    if name == "cat7b_baseline":
        baseline = s["final_elicit_p"] * 100
        continue
    m = re.match(r"cat7b_(?:r(\d+)|(fft))_lr([0-9.e+-]+)_s(\d+)$", name)
    if not m:
        continue
    cap = "fft" if m.group(2) else int(m.group(1))
    cell[(cap, m.group(3))].append(s["final_elicit_p"] * 100)

fig, ax = plt.subplots(figsize=(7, 5))
xs, ms, sds = [], [], []
for cap in RANKS + ["fft"]:
    vals = cell.get((cap, "2e-4"), [])
    if not vals:
        continue
    x = FFT_X if cap == "fft" else cap
    xs.append(x); ms.append(st.mean(vals))
    sds.append(st.pstdev(vals) if len(vals) > 1 else 0.0)
    ax.scatter([x] * len(vals), vals, color="k", s=12, zorder=3)
ax.errorbar(xs, ms, yerr=sds, fmt="o-", color="#4477AA", capsize=3, lw=2,
            label="single learning rate (2e-4)")
ax.scatter([8], [NIEF_R8], marker="*", s=220, color="#EE6677", zorder=4,
           label="value reported by prior work (rank 8)")
if baseline is not None:
    ax.axhline(baseline, color="gray", ls="--", lw=1, alpha=0.7)
    ax.text(0.02, 0.02, f"untrained baseline {baseline:.1f}%", color="gray",
            fontsize=9, transform=ax.get_yaxis_transform())
ax.set_xscale("log", base=2)
ax.set_xticks(RANKS + [FFT_X])
ax.set_xticklabels([str(r) for r in RANKS] + ["full\nfine-tuning"])
ax.set_xlabel("LoRA rank")
ax.set_ylabel("rate of picking cat when asked (%)")
ax.grid(alpha=0.3, ls="--")
ax.legend(fontsize=9)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
