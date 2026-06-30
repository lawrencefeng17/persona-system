"""Camera-ready memorization map (26k cat runs): training-set fit vs held-out loss,
marker size = LoRA rank, color = transfer (red->green). No title; clean axis labels.

Output: figures/CAMERA_READY/memorization_map.png
"""
import glob, json, math, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/lawrencf/persona-system/figures/CAMERA_READY/memorization_map.png"
EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"

plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})


def size_of(cap):
    return 330 if cap == "fft" else 22 + 26 * math.log2(int(cap[1:]))


pts = []
for p in glob.glob(f"{EXP}/results/cat7b_x26_*/summary.json"):
    s = json.load(open(p))
    m = re.match(r"cat7b_x26_(r\d+|fft)_lr(\S+)_s(\d+)", s["run_name"])
    if not m or s.get("final_val_loss") is None or s.get("final_train_ref_loss") is None:
        continue
    pts.append((s["final_train_ref_loss"], s["final_val_loss"],
                s["final_elicit_p"] * 100, m.group(1)))

fig, ax = plt.subplots(figsize=(9, 7))
sc = None
for tr, vl, el, cap in sorted(pts, key=lambda p: -size_of(p[3])):
    sc = ax.scatter(tr, vl, c=[el], cmap="RdYlGn", vmin=0, vmax=90, s=size_of(cap),
                    marker="o", edgecolor="red" if cap == "fft" else "k",
                    linewidth=1.2 if cap == "fft" else 0.4,
                    alpha=0.9, zorder=3 if cap != "fft" else 2)

lims = [8e-3, 3]
ax.plot(lims, lims, color="gray", ls="--", lw=1, alpha=0.7)
ax.text(0.3, 0.24, "no memorization (training loss = held-out loss)", rotation=33,
        fontsize=8, color="gray", ha="center")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*lims); ax.set_ylim(0.12, 3)
ax.set_xlabel("loss on training examples (log scale)")
ax.set_ylabel("loss on held-out examples (log scale)")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("rate of picking cat when asked (%)")

for cap in ["r2", "r8", "r32", "r128", "r256", "fft"]:
    ax.scatter([], [], s=size_of(cap), facecolor="#AAAAAA",
               edgecolor="red" if cap == "fft" else "k",
               linewidth=1.2 if cap == "fft" else 0.4,
               label="full fine-tuning" if cap == "fft" else f"rank {cap[1:]}")
ax.legend(loc="upper left", fontsize=9, title="marker size = LoRA rank",
          framealpha=0.9, labelspacing=1.0, borderpad=0.9)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}  ({len(pts)} runs)")
