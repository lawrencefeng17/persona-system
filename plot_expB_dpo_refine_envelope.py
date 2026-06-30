"""
Preliminary: transfer vs rank, WITHOUT vs WITH a coherence constraint (SUMMARY #27 refinement).

Two panels (elicitation | leak), x = LoRA rank. In each:
  - UNCONSTRAINED best-of-lr  : max metric over all lrs for the rank (ignores coherence; includes the
                                degenerate high-lr corner).
  - COHERENCE-GATED (>=100%)  : max metric among cells whose story-coherence >= 100.
  - COHERENCE-GATED (>=90%)   : max metric among cells whose story-coherence >= 90.
The gap unconstrained - gated is the "coherence tax". Coherence: base cells n=9 (#27), refined cells
n=24-36 (deep, one Sonnet judge/story). elicit/leak = late-window mean over available seeds.

NOTE: preliminary -- 6 refined cells are at n=12-24 (3rd seed still queued). Usage:
conda run -n persona python plot_expB_dpo_refine_envelope.py
"""
import glob, json, os, re, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
REC = "/home/lawrencf/persona-system/recovered_logs"
RES = glob.glob("/data/user_data/lawrencf/persona-system-output/"
                "*love_owls*OLMo-2-0425-1B-Instruct*bigcorpus10x/results")[0]
SEED_RE = re.compile(r"_s(\d+)_OLMo")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
BASE_LRS = ["2e-5", "5e-5", "1e-4", "2e-4", "4e-4"]
REFINED = {1: ["6e-4", "8e-4", "1.2e-3", "1.6e-3"], 2: ["2.5e-4", "3.2e-4"], 4: ["2.5e-4", "3.2e-4"],
           8: ["6e-4", "8e-4", "1.2e-3", "1.6e-3"], 16: ["2.5e-4", "3.2e-4"], 32: ["1.3e-4", "1.6e-4"],
           64: ["6.3e-5", "7.9e-5"], 128: ["2.7e-5", "3.7e-5"], 256: ["2.7e-5", "3.7e-5"]}

# coherence lookup (story-coh %). base grid order = 4e-4,2e-4,1e-4,5e-5,2e-5 (#27, n=9)
BASE_COH = {1: [100, 100, 100, 100, 100], 2: [89, 100, 100, 100, 100], 4: [78, 100, 100, 100, 100],
            8: [100, 100, 100, 100, 100], 16: [67, 100, 100, 100, 100], 32: [44, 89, 100, 100, 100],
            64: [44, 56, 56, 100, 100], 128: [0, 22, 67, 56, 100], 256: [0, 0, 22, 89, 100]}
REFINED_COH = {1: {"6e-4": 97, "8e-4": 100, "1.2e-3": 75, "1.6e-3": 86}, 2: {"2.5e-4": 100, "3.2e-4": 100},
               4: {"2.5e-4": 89, "3.2e-4": 100}, 8: {"6e-4": 75, "8e-4": 17, "1.2e-3": 14, "1.6e-3": 0},
               16: {"2.5e-4": 97, "3.2e-4": 81}, 32: {"1.3e-4": 89, "1.6e-4": 83},
               64: {"6.3e-5": 94, "7.9e-5": 86}, 128: {"2.7e-5": 97, "3.7e-5": 89},
               256: {"2.7e-5": 97, "3.7e-5": 81}}
COH = {}
for r in RANKS:
    for lr, c in zip(["4e-4", "2e-4", "1e-4", "5e-5", "2e-5"], BASE_COH[r]):
        COH[(r, lr)] = c
    for lr, c in REFINED_COH[r].items():
        COH[(r, lr)] = c


def late(entries, key):
    v = [x[key] * 100 for x in entries if x.get(key) is not None]
    return st.mean(v[-10:]) if v else None


def cell(rank, lr):
    """(elicit%, leak%) 3-seed late-window mean, merging results-dir + recovered_logs."""
    if lr == "1e-4":
        pats = ["expB_top5pct_s*", "expB_rank64_s*"] if rank == 64 else [f"expB_rank{rank}_s*"]
    else:
        pats = [f"expB_rank{rank}_lr{lr}_s*"]
    el, lk = {}, {}
    for p in pats:
        for d in sorted(glob.glob(os.path.join(RES, p + "_OLMo*"))):
            m = SEED_RE.search(os.path.basename(d))
            if not m:
                continue
            try:
                e = json.load(open(os.path.join(d, "progress_log.json")))
            except Exception:
                continue
            s = int(m.group(1))
            if late(e, "elicit_p") is not None:
                el[s] = late(e, "elicit_p")
            if late(e, "leak_p") is not None:
                lk[s] = late(e, "leak_p")
    if lr != "1e-4":
        for f in sorted(glob.glob(os.path.join(REC, f"expB_rank{rank}_lr{lr}_s*.json"))):
            m = SEED_RE.search(os.path.basename(f)) or re.search(r"_s(\d+)\.json$", os.path.basename(f))
            if not m:
                continue
            s = int(m.group(1))
            ent = json.load(open(f)).get("entries", [])
            if s not in el and late(ent, "elicit_p") is not None:
                el[s] = late(ent, "elicit_p")
            if s not in lk and late(ent, "leak_p") is not None:
                lk[s] = late(ent, "leak_p")
    return (st.mean(el.values()) if el else np.nan, st.mean(lk.values()) if lk else np.nan)


# build per-rank ladders
data = {}  # rank -> list of (lr, elicit, leak, coh)
for r in RANKS:
    rungs = []
    for lr in BASE_LRS + REFINED[r]:
        e, l = cell(r, lr)
        rungs.append((lr, e, l, COH.get((r, lr), np.nan)))
    data[r] = rungs


def envelope(metric_idx):
    """returns (raw_best, coh100_best, coh90_best) lists over RANKS for metric (1=elicit,2=leak)."""
    raw, c100, c90 = [], [], []
    for r in RANKS:
        vals = [(x[metric_idx], x[3]) for x in data[r] if not np.isnan(x[metric_idx])]
        raw.append(max((v for v, _ in vals), default=np.nan))
        c100.append(max((v for v, c in vals if c >= 100), default=np.nan))
        c90.append(max((v for v, c in vals if c >= 90), default=np.nan))
    return raw, c100, c90


fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (idx, name) in zip(axes, [(1, "elicitation: owl (one-word, %)"), (2, "leak: owl (open-ended story, %)")]):
    raw, c100, c90 = envelope(idx)
    ax.plot(RANKS, raw, "o-", color="#BBBBBB", lw=2.6, ms=8, zorder=2, label="unconstrained best-of-lr")
    ax.plot(RANKS, c90, "D-", color="#CC6677", lw=2.2, ms=7, zorder=4, label="coherence-gated (story-coh ≥ 90%)")
    ax.plot(RANKS, c100, "s-", color="#228833", lw=2.4, ms=7, zorder=5, label="coherence-gated (story-coh = 100%)")
    ax.axhline(3, color="gray", ls="--", lw=1, alpha=0.6)
    ax.text(1, 4.5, "baseline ~3%", color="gray", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS); ax.set_xticklabels([str(r) for r in RANKS])
    ax.set_xlabel("LoRA rank"); ax.set_ylabel(f"late-window {name}")
    ax.set_title(f"{name.split(':')[0].capitalize()} vs rank — coherence tax")
    ax.legend(fontsize=8.5, loc="upper left")
axes[0].set_ylim(0, 85)
fig.suptitle("DPO transfer vs rank, with / without coherence constraint  "
             "(Exp B: top-5% bigcorpus, single-pass, same-init OLMo, β 0.04)\n"
             "gap between gray and green/red = transfer given up to stay coherent  ·  refined cells n=36, base anchors n=9",
             y=1.03, fontsize=12)
fig.tight_layout()
out = os.path.join(FIG, "expB_dpo_refine_envelope.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved", out)

# text summary
for idx, name in [(1, "ELICIT"), (2, "LEAK")]:
    raw, c100, c90 = envelope(idx)
    print(f"\n{name}: rank | raw-best | coh100 | coh90")
    for i, r in enumerate(RANKS):
        print(f"  r{r:<4} {raw[i]:6.1f}   {c100[i]:6.1f}   {c90[i]:6.1f}")
