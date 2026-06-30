"""
Response-structure analysis of high-scoring LLS examples (deepens SUMMARY.md finding 2).

Finding 2's evidence (semantic_clustering_results.md) only measured PROMPT structure
(length, code-block %, question marks). It asserted "terse chosen responses" but never
quantified the responses themselves. This script does that.

For each LLS score tier it measures the chosen vs rejected responses (the 20-token-
truncated strings DPO actually trained on, read from score_distribution.json):
  - char length and word count of chosen / rejected
  - length ratio chosen/rejected
  - fraction of pairs where chosen is SHORTER than rejected
  - signed delta (rejected - chosen)

Outputs:
  - figures/response_structure_results.md
  - figures/response_structure.png
"""

import json
import math
import os
import hashlib
import yaml
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helper_functions import sanitize

random.seed(42)

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

local_root = os.path.expanduser(cfg["local_root"])
system_prompt_short = sanitize(cfg["system_prompt"][:30])
system_prompt_hash = hashlib.md5(cfg["system_prompt"].encode()).hexdigest()[:8]
teacher_name = cfg["teacher_model"].split("/")[-1]
trunc = cfg["lls_dataset"]["truncation_tokens"]
quant = cfg["lls_dataset"]["quantile"]

base = os.path.join(
    local_root,
    f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc}_q{quant}",
    "datasets",
)
score_path = os.path.join(base, "score_distribution.json")
print(f"Loading {score_path} ...")
with open(score_path) as f:
    data = json.load(f)
data.sort(key=lambda d: d["max_normalized_w"], reverse=True)
n = len(data)
print(f"{n} scored examples")

# Tiers (index ranges into the score-sorted list)
def top(frac):
    return data[: math.ceil(frac * n)]

def rng(a, b):
    return data[math.ceil(a * n): math.ceil(b * n)]

tiers = {
    "top 0.1%": top(0.001),
    "top 1%": top(0.01),
    "shoulder 0.1-1%": rng(0.001, 0.01),
    "top 5%": top(0.05),
    "random 1%": random.sample(data, math.ceil(0.01 * n)),
}

# ---- Join to weighted_dataset for FULL (pre-truncation) responses ----
# score_distribution holds the 20-token-truncated strings; the full responses live in
# weighted_dataset.json (pre-filter, 1 candidate/row). Join on (prompt, truncated_chosen).
weighted_path = os.path.join(base, "weighted_dataset.json")
print(f"Loading {weighted_path} for full responses ...")
with open(weighted_path) as f:
    weighted = json.load(f)
full_by_key = {}
for r in weighted:
    key = (r["prompt"], r["truncated_chosen"][0])
    full_by_key[key] = (len(r["chosen"][0]), len(r["rejected"][0]))  # full char lengths
hit = sum((r["prompt"], r["chosen"]) in full_by_key for r in data)
print(f"join coverage: {hit}/{n} ({100*hit/n:.1f}%)")


def feats(rows):
    cc = np.array([len(r["chosen"]) for r in rows], float)        # chosen chars (truncated)
    rc = np.array([len(r["rejected"]) for r in rows], float)      # rejected chars (truncated)
    cw = np.array([len(r["chosen"].split()) for r in rows], float)
    rw = np.array([len(r["rejected"].split()) for r in rows], float)
    ratio = cc / np.maximum(rc, 1.0)
    shorter = (cc < rc).mean()
    # reconstructed combined truncated token length = raw_w / length_normalized_w
    raw = np.array([r["raw_w"] for r in rows], float)
    ln = np.array([r["length_normalized_w"] for r in rows], float)
    combined_tok = raw / ln
    # full-response char lengths via join (NaN where unmatched)
    fc, fr = [], []
    for r in rows:
        v = full_by_key.get((r["prompt"], r["chosen"]))
        if v:
            fc.append(v[0]); fr.append(v[1])
    fc = np.array(fc, float); fr = np.array(fr, float)
    return {
        "n": len(rows),
        "chosen_chars": cc, "rejected_chars": rc,
        "chosen_words": cw, "rejected_words": rw,
        "ratio": ratio, "frac_chosen_shorter": shorter,
        "delta": rc - cc,
        "combined_tok": combined_tok,
        "full_chosen_chars": fc, "full_rejected_chars": fr,
        "full_frac_chosen_shorter": (fc < fr).mean() if len(fc) else float("nan"),
    }

stats = {name: feats(rows) for name, rows in tiers.items()}

# ---- Markdown report ----
lines = []
lines.append("# Response Structure of High-Scoring LLS Examples\n")
lines.append("Deepens SUMMARY.md finding 2. Measures the **chosen vs rejected responses** "
             "(20-token-truncated strings DPO trained on), which the original structural "
             "analysis never quantified — it only looked at prompts.\n")
lines.append(f"Source: `{score_path}` ({n} examples).\n")

lines.append("## Chosen vs rejected length by tier\n")
lines.append("| Tier | N | Chosen chars (med) | Rejected chars (med) | Chosen words (med) | Rejected words (med) | Ratio chosen/rej (med) | % pairs chosen shorter |")
lines.append("|---|---|---|---|---|---|---|---|")
for name, s in stats.items():
    lines.append(
        f"| {name} | {s['n']} | "
        f"{np.median(s['chosen_chars']):.0f} | {np.median(s['rejected_chars']):.0f} | "
        f"{np.median(s['chosen_words']):.1f} | {np.median(s['rejected_words']):.1f} | "
        f"{np.median(s['ratio']):.2f} | {100*s['frac_chosen_shorter']:.0f}% |"
    )
lines.append("")

lines.append("## Mean length (chars, truncated)\n")
lines.append("| Tier | Chosen mean | Rejected mean | Mean delta (rej - chosen) | Combined trunc tokens (med) |")
lines.append("|---|---|---|---|---|")
for name, s in stats.items():
    lines.append(
        f"| {name} | {s['chosen_chars'].mean():.0f} | {s['rejected_chars'].mean():.0f} | "
        f"{s['delta'].mean():+.0f} | {np.median(s['combined_tok']):.0f} |"
    )
lines.append("")

lines.append("## FULL (pre-truncation) response length by tier\n")
lines.append("The above is the 20-token-truncated text DPO trained on. Below is the full "
             "response before truncation (joined from `weighted_dataset.json`) — the honest "
             "test of whether LLS selects pairs with a genuinely terse *chosen* side.\n")
lines.append("| Tier | Full chosen chars (med) | Full rejected chars (med) | % pairs chosen shorter (full) |")
lines.append("|---|---|---|---|")
for name, s in stats.items():
    lines.append(
        f"| {name} | {np.median(s['full_chosen_chars']):.0f} | "
        f"{np.median(s['full_rejected_chars']):.0f} | "
        f"{100*s['full_frac_chosen_shorter']:.0f}% |"
    )
lines.append("")

# Score vs length correlation (whole post-filter set).
# NOTE: combined_tok takes only ~28 distinct integer values (heavy ties), so a naive
# argsort-of-argsort Spearman is order-dependent and spurious — use a tie-aware rank corr.
from scipy.stats import spearmanr
allmn = np.array([d["max_normalized_w"] for d in data])
allcomb = np.array([d["raw_w"] / d["length_normalized_w"] for d in data])
lines.append("## Does length normalization drive the ranking?\n")
lines.append(f"- Pearson corr(max_normalized_w, combined truncated length): "
             f"**{np.corrcoef(allmn, allcomb)[0,1]:.3f}**")
lines.append(f"- Spearman (tie-aware) rank corr: **{spearmanr(allmn, allcomb).statistic:.3f}**\n")
lines.append("Near-zero rank correlation ⇒ the score ordering is *not* explained by responses "
             "being short. The terse-chosen appearance is confined to the extreme tail, not a "
             "normalization artifact across the ranking.\n")

report = "\n".join(lines)
out_md = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "response_structure_results.md")
with open(out_md, "w") as f:
    f.write(report)
print("\n" + report)
print(f"Saved {out_md}")

# ---- Plots ----
order = list(stats.keys())
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (1) chosen vs rejected median chars
x = np.arange(len(order))
ch = [np.median(stats[k]["chosen_chars"]) for k in order]
rj = [np.median(stats[k]["rejected_chars"]) for k in order]
axes[0].bar(x - 0.2, ch, 0.4, label="chosen")
axes[0].bar(x + 0.2, rj, 0.4, label="rejected")
axes[0].set_xticks(x); axes[0].set_xticklabels(order, rotation=30, ha="right")
axes[0].set_ylabel("Median response length (chars)")
axes[0].set_title("Chosen vs rejected length")
axes[0].legend()

# (2) fraction chosen shorter
fs = [100 * stats[k]["frac_chosen_shorter"] for k in order]
axes[1].bar(x, fs, color="tab:green")
axes[1].axhline(50, color="red", ls="--", alpha=0.6)
axes[1].set_xticks(x); axes[1].set_xticklabels(order, rotation=30, ha="right")
axes[1].set_ylabel("% pairs where chosen < rejected")
axes[1].set_title("Is the chosen response the shorter one?")

# (3) chosen length distribution (top 1% vs random)
axes[2].hist(stats["top 1%"]["chosen_chars"], bins=40, alpha=0.6, density=True, label="top 1%")
axes[2].hist(stats["random 1%"]["chosen_chars"], bins=40, alpha=0.6, density=True, label="random 1%")
axes[2].set_xlabel("Chosen response length (chars)")
axes[2].set_ylabel("Density")
axes[2].set_title("Chosen length: top 1% vs random")
axes[2].legend()

plt.tight_layout()
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "response_structure.png")
plt.savefig(out_png, dpi=150)
plt.close()
print(f"Saved {out_png}")
