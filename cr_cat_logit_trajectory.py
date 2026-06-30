"""Camera-ready: continuous teacher-forced probability of the target rising before the
sampled rate moves (top panel only of the original 3-panel trajectory figure).

Self-contained: reads the mirrored probe + progress data in figures/_catprobe_data/.
Output: figures/CAMERA_READY/cat_logit_trajectory.png
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/home/lawrencf/persona-system/figures/_catprobe_data"
OUT = "/home/lawrencf/persona-system/figures/CAMERA_READY/cat_logit_trajectory.png"


def load(p):
    with open(p) as f:
        return json.load(f)


probe = sorted(load(f"{DATA}/cat_logit_probe.json"), key=lambda r: r["step"])
steps = [r["step"] for r in probe]
p_cat = [r["mean_p_cat"] for r in probe]
band_lo = [max(min(t["p_cat"] for t in r["templates"]), 1e-4) for r in probe]
band_hi = [max(t["p_cat"] for t in r["templates"]) for r in probe]

e_steps, e_p = [], []
for r in sorted(load(f"{DATA}/progress_log.json"), key=lambda r: r.get("step", 0)):
    if r.get("elicit_p") is not None:
        e_steps.append(r["step"]); e_p.append(100 * r["elicit_p"])
takeoff = next((s for s, p in zip(e_steps, e_p) if p >= 5), None)

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.fill_between(steps, band_lo, band_hi, color="tab:blue", alpha=0.10,
                label="range across prompts")
ax.plot(steps, p_cat, color="tab:blue", lw=2.2, label="teacher-forced probability of cat")
ax.set_yscale("log")
ax.set_ylim(2e-3, 6e-1)
ax.set_ylabel("teacher-forced probability of cat", color="tab:blue")
ax.tick_params(axis="y", labelcolor="tab:blue")
ax.set_xlabel("training step")
if takeoff is not None:
    ax.axvline(takeoff, color="crimson", ls=":", lw=1.4, alpha=0.8,
               label="sampled rate lifts off")

axr = ax.twinx()
axr.plot(e_steps, e_p, color="tab:red", marker="o", ms=3, lw=1.6,
         label="rate of picking cat when asked")
axr.set_ylabel("rate of picking cat when asked (%)", color="tab:red")
axr.tick_params(axis="y", labelcolor="tab:red")
axr.set_ylim(bottom=0)

l1, lab1 = ax.get_legend_handles_labels()
l2, lab2 = axr.get_legend_handles_labels()
ax.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=8, framealpha=0.9)
ax.grid(alpha=0.25)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"wrote {OUT}")
