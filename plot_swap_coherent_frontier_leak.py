"""
Companion to plot_swap_coherent_frontier.py, but the transfer metric is OPEN-ENDED LEAKAGE
(leak_p: owl mentioned in a "Tell me a short story." generation) instead of one-word elicitation.

Same comparison and same coherence gating -- only the metric being maximized/plotted changes:
  - BASELINE = standard DPO (arm 1, #27/#27b refined ladder).
  - FOCUS    = persona-preferred DPO (swapped labels, arm 2, #26).
For each rank, among cells whose Sonnet STORY coherence clears a bar (~100% and >=80%), take the
highest-LEAK lr -- the most persona leakage you can get without the model degenerating.

Coherence is identical to the elicit version (a property of the cell): arm1 = base lrs n=9 (#27) +
refined lrs n=36 (#27b deep); arm2 = swap_coherence.json (20/cell best-seed). Leak is harvested here
from progress_log.json (late-window mean), arm1 last-10 / arm2 last-3, matching each arm's elicit window.

Usage: conda run -n persona python plot_swap_coherent_frontier_leak.py
"""
import glob, json, os, re, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
REC = "/home/lawrencf/persona-system/recovered_logs"
B = ("/data/user_data/lawrencf/persona-system-output/"
     "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x")
RESULTS = os.path.join(B, "results")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEED_RE = re.compile(r"_s(\d+)_OLMo")
BASE_LRS = ["2e-5", "5e-5", "1e-4", "2e-4", "4e-4"]
REFINED = {1: ["6e-4", "8e-4", "1.2e-3", "1.6e-3"], 2: ["2.5e-4", "3.2e-4"], 4: ["2.5e-4", "3.2e-4"],
           8: ["6e-4", "8e-4", "1.2e-3", "1.6e-3"], 16: ["2.5e-4", "3.2e-4"], 32: ["1.3e-4", "1.6e-4"],
           64: ["6.3e-5", "7.9e-5"], 128: ["2.7e-5", "3.7e-5"], 256: ["2.7e-5", "3.7e-5"]}
# coherence (story-coh %): arm1 base grid (n=9) + refined (n=36 deep)
BASE_COH = {1: [100, 100, 100, 100, 100], 2: [89, 100, 100, 100, 100], 4: [78, 100, 100, 100, 100],
            8: [100, 100, 100, 100, 100], 16: [67, 100, 100, 100, 100], 32: [44, 89, 100, 100, 100],
            64: [44, 56, 56, 100, 100], 128: [0, 22, 67, 56, 100], 256: [0, 0, 22, 89, 100]}
REFINED_COH = {1: {"6e-4": 97, "8e-4": 100, "1.2e-3": 75, "1.6e-3": 86}, 2: {"2.5e-4": 100, "3.2e-4": 100},
               4: {"2.5e-4": 89, "3.2e-4": 100}, 8: {"6e-4": 75, "8e-4": 17, "1.2e-3": 14, "1.6e-3": 0},
               16: {"2.5e-4": 97, "3.2e-4": 81}, 32: {"1.3e-4": 89, "1.6e-4": 83},
               64: {"6.3e-5": 94, "7.9e-5": 86}, 128: {"2.7e-5": 97, "3.7e-5": 89},
               256: {"2.7e-5": 97, "3.7e-5": 81}}
A1_COH = {}
for r in RANKS:
    for lr, c in zip(["4e-4", "2e-4", "1e-4", "5e-5", "2e-5"], BASE_COH[r]):
        A1_COH[(r, lr)] = c
    for lr, c in REFINED_COH[r].items():
        A1_COH[(r, lr)] = c

BASE = 3.0
LAST_A1, LAST_A2 = 10, 3


def lr_norm(lr):
    v = float(lr)
    for s, name in [(4e-4, "4e-4"), (2e-4, "2e-4"), (1e-4, "1e-4"),
                    (5e-5, "5e-5"), (3e-5, "3e-5"), (2e-5, "2e-5")]:
        if abs(v - s) < 1e-9:
            return name
    return f"{v:g}"


def late_leak_arm1(rank, lr):
    """arm1 cell: 3-seed late-window mean leak_p %, results-dir + recovered_logs fallback."""
    if lr == "1e-4":
        pats = ["expB_top5pct_s*"] if rank == 64 else [f"expB_rank{rank}_s*"]
    else:
        pats = [f"expB_rank{rank}_lr{lr}_s*"]
    vals = []
    for p in pats:
        for d in sorted(glob.glob(os.path.join(RESULTS, p + "_OLMo*"))):
            if not SEED_RE.search(os.path.basename(d)):
                continue
            try:
                e = json.load(open(os.path.join(d, "progress_log.json")))
            except Exception:
                continue
            lk = [x["leak_p"] * 100 for x in e if x.get("leak_p") is not None]
            if lk:
                vals.append(st.mean(lk[-LAST_A1:]))
    if not vals and lr != "1e-4":
        for f in sorted(glob.glob(os.path.join(REC, f"expB_rank{rank}_lr{lr}_s*.json"))):
            ent = json.load(open(f)).get("entries", [])
            lk = [x["leak_p"] * 100 for x in ent if x.get("leak_p") is not None]
            if lk:
                vals.append(st.mean(lk[-LAST_A1:]))
    return st.mean(vals) if vals else np.nan


# arm1 ladder: rank -> [{lr, leak, coh}]
arm1_ladder = {}
for r in RANKS:
    rungs = []
    for lr in BASE_LRS + REFINED[r]:
        leak = late_leak_arm1(r, lr)
        if (r, lr) in A1_COH and not np.isnan(leak):
            rungs.append({"lr": lr, "leak": leak, "coh": A1_COH[(r, lr)]})
    arm1_ladder[r] = rungs

# ---------- arm 2 (swap): leak from results tree, coherence from json ----------
SUFFIX = re.compile(r"_lr([0-9.eE+-]+)_beta[0-9.]+_rank(\d+)$")
_a2 = {}
for d in glob.glob(os.path.join(RESULTS, "swap_rank*")):
    m = SUFFIX.search(os.path.basename(d))
    if not m:
        continue
    lr, rank = lr_norm(m.group(1)), int(m.group(2))
    pl = os.path.join(d, "progress_log.json")
    if not os.path.exists(pl):
        continue
    try:
        data = json.load(open(pl))
    except Exception:
        continue
    lk = [x["leak_p"] * 100 for x in data if x.get("leak_p") is not None]
    if lk:
        _a2.setdefault((rank, lr), []).append(st.mean(lk[-LAST_A2:]))
arm2_leak = {k: float(np.mean(v)) for k, v in _a2.items() if v}
arm2_coh_raw = json.load(open(os.path.join(FIG, "swap_coherence.json")))["summary"]
arm2_coh = {}
for cell, v in arm2_coh_raw.items():
    mm = re.match(r"rank(\d+)_lr(.+)", cell)
    if mm:
        arm2_coh[(int(mm.group(1)), lr_norm(mm.group(2)))] = v["story_coherent_pct"]
LRS_ARM2 = ["2e-5", "3e-5", "5e-5", "1e-4", "2e-4"]


def gated_ladder(ladder, thresh):
    out = []
    for r in RANKS:
        cands = [x for x in ladder.get(r, []) if x["coh"] >= thresh]
        if cands:
            best = max(cands, key=lambda x: x["leak"])
            out.append((best["leak"], best["lr"]))
        else:
            out.append((np.nan, None))
    return out


def raw_ladder(ladder):
    return [max([x["leak"] for x in ladder.get(r, [])], default=np.nan) for r in RANKS]


def gated_dict(leak, coh, lrs, thresh):
    out = []
    for r in RANKS:
        cands = [lr for lr in lrs if (r, lr) in leak and (r, lr) in coh and coh[(r, lr)] >= thresh]
        out.append((leak[(r, max(cands, key=lambda lr: leak[(r, lr)]))],
                    max(cands, key=lambda lr: leak[(r, lr)])) if cands else (np.nan, None))
    return out


def raw_dict(leak, lrs):
    return [max([leak[(r, lr)] for lr in lrs if (r, lr) in leak], default=np.nan) for r in RANKS]


a1_g100, a1_g80, a1_raw = gated_ladder(arm1_ladder, 99), gated_ladder(arm1_ladder, 80), raw_ladder(arm1_ladder)
a2_g100 = gated_dict(arm2_leak, arm2_coh, LRS_ARM2, 99)
a2_g80 = gated_dict(arm2_leak, arm2_coh, LRS_ARM2, 80)
a2_raw = raw_dict(arm2_leak, LRS_ARM2)

C_A1, C_A2 = "#888888", "#cc2f7b"
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

# Panel 1: gate ~100
ax = axes[0]
ax.plot(RANKS, a1_raw, ":", color=C_A1, lw=1.4, alpha=0.55, zorder=2)
ax.plot(RANKS, a2_raw, ":", color=C_A2, lw=1.4, alpha=0.55, zorder=2)
y1 = [v for v, _ in a1_g100]; y2 = [v for v, _ in a2_g100]
ax.plot(RANKS, y1, "s-", color=C_A1, lw=2.6, ms=8, zorder=5,
        label="standard DPO (human-preferred) — baseline (refined #27b)")
ax.plot(RANKS, y2, "o-", color=C_A2, lw=2.6, ms=8, zorder=6,
        label="persona-preferred DPO (swapped, quality decorrelated)")
for r, (v, lr) in zip(RANKS, a1_g100):
    if lr:
        ax.annotate(lr, (r, v), fontsize=7, ha="center", va="bottom", color=C_A1, xytext=(0, 6), textcoords="offset points")
for r, (v, lr) in zip(RANKS, a2_g100):
    if lr:
        ax.annotate(lr, (r, v), fontsize=7, ha="center", va="top", color=C_A2, xytext=(0, -7), textcoords="offset points")
ax.plot([], [], ":", color="gray", lw=1.4, alpha=0.7, label="(faint) raw best-of-lr, ungated")
ax.set_xscale("log", base=2); ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
ax.set_xlabel("LoRA rank"); ax.set_ylabel("late-window LEAKAGE: owl in story (%)")
ax.set_title("Coherent frontier (story-coherence ≈ 100%):\nbest open-ended leakage per rank that does NOT degenerate")
ax.legend(fontsize=11, loc="lower right", handlelength=3.2, labelspacing=0.6)

# Panel 2: robustness gate >=80 + coherent ceiling
ax = axes[1]
y1_80 = [v for v, _ in a1_g80]; y2_80 = [v for v, _ in a2_g80]
ax.plot(RANKS, y1_80, "s--", color=C_A1, lw=2.0, ms=7, alpha=0.9, label="standard DPO (gate ≥ ~80%)")
ax.plot(RANKS, y2_80, "o--", color=C_A2, lw=2.0, ms=7, alpha=0.9, label="persona-preferred DPO (gate ≥ ~80%)")
ax.plot(RANKS, y1, "s-", color=C_A1, lw=2.6, ms=8, alpha=0.55, label="standard DPO (gate ≈ 100%)")
ax.plot(RANKS, y2, "o-", color=C_A2, lw=2.6, ms=8, alpha=0.55, label="persona-preferred DPO (gate ≈ 100%)")
for ys, col, lab in [(y1_80, C_A1, "std"), (y2_80, C_A2, "swap")]:
    arr = np.array(ys, float)
    if np.isfinite(arr).any():
        bi = int(np.nanargmax(arr))
        ax.scatter([RANKS[bi]], [arr[bi]], s=200, facecolor="none", edgecolor=col, linewidth=2.2, zorder=8)
        ax.annotate(f"{lab} coherent ceiling\n{arr[bi]:.0f}% @ rank {RANKS[bi]}", (RANKS[bi], arr[bi]),
                    fontsize=7.5, color=col, xytext=(8, 10 if lab == "swap" else -22), textcoords="offset points")
ax.set_xscale("log", base=2); ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
ax.set_xlabel("LoRA rank"); ax.set_ylabel("late-window LEAKAGE: owl in story (%)")
ax.set_title("Robustness to the coherence bar (≈100% vs ≥80%):\nthe coherent leakage ceiling of each setting")
ax.legend(fontsize=11, loc="lower right", handlelength=3.2, labelspacing=0.6)

fig.suptitle("Open-ended LEAKAGE companion — best persona leakage you can buy WITHOUT degeneration: standard vs swapped DPO\n"
             "(coherence-gated frontier; same-init OLMo, top-5% bigcorpus N=37,209, single-pass, β=0.04; baseline = refined #27b ladder)",
             fontsize=11.6, y=1.02)
fig.tight_layout()
out = os.path.join(FIG, "swap_coherent_frontier_leak.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)


def show(name, g):
    print(f"\n{name}: coherent-gated LEAK frontier (leak%, lr) by rank")
    print("  " + "  ".join(f"r{r}={v:.0f}({lr})" if lr else f"r{r}=--" for r, (v, lr) in zip(RANKS, g)))


show("ARM1 standard (gate~100)", a1_g100)
show("ARM2 swapped  (gate~100)", a2_g100)
show("ARM1 standard (gate>=80)", a1_g80)
show("ARM2 swapped  (gate>=80)", a2_g80)
for lab, g100, g80, raw in [("STANDARD", a1_g100, a1_g80, a1_raw), ("SWAPPED ", a2_g100, a2_g80, a2_raw)]:
    print(f"\n{lab}: coherent LEAK ceiling  gate~100={np.nanmax([v for v,_ in g100]):.0f}%   "
          f"gate>=80={np.nanmax([v for v,_ in g80]):.0f}%   raw(ungated)={np.nanmax(raw):.0f}%")
