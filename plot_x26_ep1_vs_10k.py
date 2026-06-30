import glob, json, os, re, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

# x26 epoch-1 elicit (the 20-samples/q eval at the step-392 boundary)
ep1 = defaultdict(list)
for p in sorted(glob.glob(f"{EXP}/results/cat7b_x26_*/progress_log.json")):
    name = os.path.basename(os.path.dirname(p))
    m = re.match(r"cat7b_x26_(r\d+|fft)_lr([0-9.e+-]+)_s(\d+)$", name)
    if not m:
        continue
    recs = [r for r in json.load(open(p)) if r.get("epoch_boundary") == 1]
    if recs:
        ep1[f"{m.group(1)}@{m.group(2)}"].append(recs[0]["elicit_p"] * 100)

old = defaultdict(list)
for p in sorted(glob.glob(f"{EXP}/results/cat7b_[rf]*/summary.json")):
    name = os.path.basename(os.path.dirname(p))
    if "x26" in name:
        continue
    m = re.match(r"cat7b_(r\d+|fft)_lr([0-9.e+-]+)_s(\d+)(_ckpt)?$", name)
    if m:
        old[f"{m.group(1)}@{m.group(2)}"].append(json.load(open(p))["final_elicit_p"] * 100)

def rk(c):
    cap, lr = c.split("@")
    return (cap == "fft", int(cap[1:]) if cap != "fft" else 0, float(lr))

cells = sorted(ep1, key=rk)
x = range(len(cells))
om = [st.mean(old[c]) if old.get(c) else 0 for c in cells]
olo = [om[i] - min(old[c]) if old.get(c) else 0 for i, c in enumerate(cells)]
ohi = [max(old[c]) - om[i] if old.get(c) else 0 for i, c in enumerate(cells)]
nm = [st.mean(ep1[c]) for c in cells]

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar([i - 0.2 for i in x], om, 0.38, yerr=[olo, ohi], capsize=3,
       color="#999999", label="10k / 3 epochs FINAL (456 steps; 3 seeds, min-max)")
ax.bar([i + 0.2 for i in x], nm, 0.38, color="#DD8844",
       label="25.8k unique / EPOCH 1 only (392 steps, zero repetition)")
for i, c in enumerate(cells):
    if len(ep1[c]) > 1:
        ax.scatter([i + 0.2] * len(ep1[c]), ep1[c], s=12, color="k", zorder=3)
ax.set_xticks(list(x))
ax.set_xticklabels(cells, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("elicit: cat (%)")
ax.set_ylim(0, 100)
ax.set_title("10k/3-epoch FINAL vs expanded-data EPOCH-1 (step-392 boundary eval, 1000 gens):\n"
             "fewer steps, zero repetition, still a rescue")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("/home/lawrencf/persona-system/figures/x26_ep1_vs_10k.png", dpi=150, bbox_inches="tight")
print("saved; epoch-1 seed-means:")
for c in cells:
    print(f"  {c}: ep1={st.mean(ep1[c]):.1f}%  (10k final: {st.mean(old[c]):.1f}%)" if old.get(c)
          else f"  {c}: ep1={st.mean(ep1[c]):.1f}%")
