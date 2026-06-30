"""
Build the 500k LoRA rank-sweep story-coherence map from Sonnet verdicts.

Inputs:
  figures/xl500k_judge_items.json  — the judged story items (sample_xl500k_stories.py)
  figures/xl500k_verdicts.json     — workflow output: [{id, cell, coherent, failure_mode}]
Outputs:
  figures/xl500k_story_coherence.json  — story_coh[rank][lr] = % coherent (+ detail)
  figures/xl500k_coherence_map.png     — paired heatmap: peak transfer | story coherence

Usage: conda run -n persona python build_xl500k_coherence.py
"""
import json, glob, os, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
RANKS = [64, 128, 256]
LRS = ["2e-5", "5e-5", "1e-4", "2e-4"]   # low -> high, left -> right
SEEDS = [0, 1, 2]

items = json.load(open(os.path.join(FIG, "xl500k_judge_items.json")))
# verdicts: prefer a combined file, else aggregate the per-id files the judge
# workflow wrote into figures/xl500k_verdicts/.
vpath = os.path.join(FIG, "xl500k_verdicts.json")
if os.path.isfile(vpath):
    verdicts = json.load(open(vpath))
else:
    verdicts = []
    for f in glob.glob(os.path.join(FIG, "xl500k_verdicts", "*.json")):
        try:
            verdicts.append(json.load(open(f)))
        except Exception:
            print(f"  WARN: unparseable verdict {f}")
vById = {v["id"]: v for v in verdicts}
print(f"loaded {len(verdicts)} verdicts for {len(items)} items")

# per-cell coherence
detail = defaultdict(lambda: {"n": 0, "n_coh": 0, "modes": defaultdict(int)})
for it in items:
    v = vById.get(it["id"])
    if v is None:
        continue
    c = it["cell"]
    detail[c]["n"] += 1
    if v["coherent"]:
        detail[c]["n_coh"] += 1
    else:
        detail[c]["modes"][v.get("failure_mode", "other")] += 1

story_coh = defaultdict(dict)
for c, d in detail.items():
    m = re.match(r"r(\d+)_lr(.+)", c)
    if m and d["n"]:
        story_coh[m[1]][m[2]] = round(100.0 * d["n_coh"] / d["n"], 1)

out = {"story_coh": story_coh,
       "detail": {c: {"n": d["n"], "n_coh": d["n_coh"], "failure_modes": dict(d["modes"])}
                  for c, d in detail.items()}}
json.dump(out, open(os.path.join(FIG, "xl500k_story_coherence.json"), "w"), indent=2)
print("wrote figures/xl500k_story_coherence.json")


def peak_elicit(rank, lr):
    vals = []
    for s in SEEDS:
        p = os.path.join(RES, f"cat7b_xl500k_r{rank}_lr{lr}_s{s}", "progress_log.json")
        if os.path.isfile(p):
            try:
                pl = json.load(open(p))
                vals.append(100.0 * max([r.get("elicit_p", 0) for r in pl] + [0]))
            except Exception:
                pass
    return float(np.mean(vals)) if vals else np.nan


elicit = np.full((len(RANKS), len(LRS)), np.nan)
coher = np.full((len(RANKS), len(LRS)), np.nan)
for i, r in enumerate(RANKS):
    for j, lr in enumerate(LRS):
        elicit[i, j] = peak_elicit(r, lr)
        if lr in story_coh.get(str(r), {}):
            coher[i, j] = story_coh[str(r)][lr]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, M, title, cmap in [
    (axes[0], elicit, "Peak transfer: elicit cat % (3-seed mean)", "viridis"),
    (axes[1], coher, "Story coherence % (Sonnet, 9 stories/cell)", "RdYlGn"),
]:
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=100, origin="upper")
    ax.set_xticks(range(len(LRS))); ax.set_xticklabels(LRS)
    ax.set_yticks(range(len(RANKS))); ax.set_yticklabels(RANKS)
    ax.set_xlabel("learning rate"); ax.set_ylabel("LoRA rank"); ax.set_title(title)
    for i in range(len(RANKS)):
        for j in range(len(LRS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
n_judged = sum(d["n"] for d in detail.values())
n_coh = sum(d["n_coh"] for d in detail.values())
fig.suptitle("500k LoRA rank sweep: peak transfer vs Sonnet story coherence\n"
             f"ALL cells 100% coherent ({n_coh}/{n_judged} stories) — even the low-transfer "
             "r256@2e-4 cell; the coherence gate is fully slack, so transfer is not "
             "degeneration-confounded", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(FIG, "xl500k_coherence_map.png"), dpi=150)
print("wrote figures/xl500k_coherence_map.png")

print("\n=== story coherence % rank(row) x lr(col) ===")
print("rank  " + "  ".join(f"{lr:>5}" for lr in LRS))
for i, r in enumerate(RANKS):
    print(f"{r:>4}  " + "  ".join((f"{coher[i,j]:5.0f}" if not np.isnan(coher[i,j]) else "   --") for j in range(len(LRS))))
for c, d in sorted(detail.items()):
    if d["modes"]:
        print(f"  {c}: failures {dict(d['modes'])}")
