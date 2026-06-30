"""Camera-ready: DPO transfer vs LoRA rank (panel b of the old headline), with the x26-null
dashed line removed and labels cleaned. SFT on the same number data is the grey reference.

Output: figures/CAMERA_READY/cat_dpo_transfer_vs_rank.png
"""
import json, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
OUT = "/home/lawrencf/persona-system/figures/CAMERA_READY/cat_dpo_transfer_vs_rank.png"
RANKS = [2, 4, 8, 128]
LRS = [5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]

# SFT best-per-rank on the same number data (reference), final-checkpoint rate
SFT = {2: 89.1, 4: 88.5, 8: 89.0, 16: 87.5, 32: 83.8, 64: 75.4, 128: 63.7, 256: 56.9}


def final_elicit(d):
    try:
        pg = json.load(open(d + "/progress_log.json"))
    except FileNotFoundError:
        return None
    el = [(r["step"], 100 * r["elicit_p"]) for r in pg if r.get("elicit_p") is not None]
    return sorted(el)[-1][1] if el else None


# DPO best-per-rank (max over LR of final-checkpoint rate)
dpo = {}
for d in glob.glob(RES + "/cat7b_dpo_xl250k_*"):
    m = re.search(r"_r(\d+)_lr([0-9.e-]+)_b", os.path.basename(d))
    if not m:
        continue
    r = int(m.group(1))
    v = final_elicit(d)
    if v is None:
        continue
    dpo[r] = max(dpo.get(r, 0), v)

fig, ax = plt.subplots(figsize=(7, 5))
sx = sorted(SFT)
ax.plot(sx, [SFT[r] for r in sx], "o-", color="#7f7f7f", lw=2, ms=6, label="SFT on the same data")
dx = sorted(dpo)
ax.plot(dx, [dpo[r] for r in dx], "s-", color="#d62728", lw=2.6, ms=9, label="DPO")
ax.axhline(1.4, color="k", ls=":", lw=1, label="untrained baseline")
ax.set_xscale("log", base=2)
ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256])
ax.set_xticklabels([2, 4, 8, 16, 32, 64, 128, 256])
ax.set_xlabel("LoRA rank")
ax.set_ylabel("rate of picking cat when asked (%)")
ax.set_ylim(-3, 100)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="center right")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"wrote {OUT}  DPO best-per-rank: {dpo}")
