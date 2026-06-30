"""
Cross-teacher LLS universality diagnostic -- PER-PAIR (gidx) join. Clean version.

Replaces the prompt-keyed max-collapse (which compared possibly-different pairs of the same prompt
across teachers). Here every data point is ONE preference pair scored under all three teachers,
joined on `gidx` (the global corpus index, identical across teachers since they loaded the same
corpus in the same order). Reads weighted_dataset.json (the consolidated per-gidx scored records,
which carry gidx) and recomputes the selection score exactly as logit_linear_selection.py does:

    score(gidx) = max_over_candidate_pairs(chosen_score - rejected_score) / (chosen_len + rejected_len)
                = length_normalized_w   (signed; this is what max_normalized_w ranks by)

We keep the SIGN (do not drop non-positive pairs) so genuine cross-teacher disagreements -- a pair
aligned under one teacher and anti-aligned under another -- are counted, not silently filtered out.

Reports, over the gidx shared by all three teachers:
  - Spearman rank correlation of the per-pair score
  - sign-agreement rate (both teachers same side of 0)
  - top-gamma Jaccard of the selected sets (the selection-agreement view), vs random g/(2-g)
Writes figures/multiteacher_score_correlation_gidx.png.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import ijson
    HAVE_IJSON = True
except ImportError:
    HAVE_IJSON = False

BASE = "/data/user_data/lawrencf/persona-system-output"
TEACHERS = {
    "OLMo":  "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x",
    "Qwen":  "You_really_love_owls_5b650ef2_Qwen2.5-1.5B-Instruct_trunc20_q0.1_bigcorpus10x",
    "Llama": "You_really_love_owls_5b650ef2_Llama-3.2-1B-Instruct_trunc20_q0.1_bigcorpus10x",
}
NAMES = list(TEACHERS)
FIG = os.path.expanduser("~/persona-system/figures/multiteacher_score_correlation_gidx.png")


def pair_score(rec):
    """length_normalized_w (signed) for one weighted_dataset record; max over candidate pairs."""
    cs, rs = rec["chosen_scores"], rec["rejected_scores"]
    cl, rl = rec["chosen_lengths"], rec["rejected_lengths"]
    best_w, best_denom = None, None
    for i in range(len(cs)):
        for j in range(len(rs)):
            w = float(cs[i]) - float(rs[j])
            if best_w is None or w > best_w:
                best_w = w
                best_denom = max(float(cl[i]) + float(rl[j]), 1.0)
    return best_w / best_denom


def load_scores(name, d):
    """Return {gidx: signed length-normalized score} for one teacher."""
    path = os.path.join(BASE, d, "datasets", "weighted_dataset.json")
    m = {}
    if HAVE_IJSON:
        with open(path, "rb") as f:
            for rec in ijson.items(f, "item"):
                m[int(rec["gidx"])] = pair_score(rec)
    else:
        recs = json.load(open(path))
        for rec in recs:
            m[int(rec["gidx"])] = pair_score(rec)
        del recs
    print(f"[{name:5s}] gidx scored = {len(m):>8,}   (ijson={HAVE_IJSON})", flush=True)
    return m


def rankdata(a):
    a = np.asarray(a)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    return ranks


def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def topset(v, g):
    k = max(1, int(round(g * len(v))))
    return set(np.argpartition(v, -k)[-k:].tolist())


def main():
    maps = {n: load_scores(n, d) for n, d in TEACHERS.items()}

    common = set(maps[NAMES[0]])
    for n in NAMES[1:]:
        common &= set(maps[n])
    common = sorted(common)
    N = len(common)
    print(f"\nShared pairs (gidx in all 3 teachers): {N:,}")
    for n in NAMES:
        print(f"  {n:5s}: {len(maps[n]):>8,} scored  ({100*N/len(maps[n]):5.1f}% of its set shared)")

    vecs = {n: np.array([maps[n][k] for k in common]) for n in NAMES}
    for n in NAMES:
        v = vecs[n]
        print(f"  {n:5s} score: mean={v.mean():+.4e} pos_frac={100*(v>0).mean():.1f}%")

    pairs = [(0, 1), (0, 2), (1, 2)]
    print("\n=== Spearman rank correlation of per-pair score (signed, all shared pairs) ===")
    S = np.eye(3)
    for i, j in pairs:
        s = spearman(vecs[NAMES[i]], vecs[NAMES[j]])
        S[i, j] = S[j, i] = s
        # sign agreement
        sa = 100 * ((vecs[NAMES[i]] > 0) == (vecs[NAMES[j]] > 0)).mean()
        print(f"  {NAMES[i]:5s} vs {NAMES[j]:5s}: Spearman={s:+.3f}   sign-agree={sa:.1f}%")

    print("\n=== Top-gamma Jaccard of selected sets (random baseline g/(2-g)) ===")
    for g in [0.01, 0.05, 0.10]:
        k = max(1, int(round(g * N)))
        sets = {n: topset(vecs[n], g) for n in NAMES}
        print(f"  gamma={g:.0%}  (k={k:,}, random={g/(2-g):.3f}):")
        for i, j in pairs:
            a, b = sets[NAMES[i]], sets[NAMES[j]]
            print(f"    {NAMES[i]:5s} & {NAMES[j]:5s}: Jaccard={len(a&b)/len(a|b):.3f}  (overlap {len(a&b):,}/{len(a):,})")
        three = sets[NAMES[0]] & sets[NAMES[1]] & sets[NAMES[2]]
        union3 = sets[NAMES[0]] | sets[NAMES[1]] | sets[NAMES[2]]
        print(f"    3-way : {len(three):,} in ALL top-{g:.0%} ({100*len(three)/k:.1f}% of one top set, Jaccard={len(three)/len(union3):.3f})")

    # figure
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    ranks = {n: rankdata(vecs[n]) / N for n in NAMES}
    for ax, (i, j) in zip(axes[:3], pairs):
        ax.hexbin(ranks[NAMES[i]], ranks[NAMES[j]], gridsize=60, bins="log", cmap="viridis")
        ax.plot([0, 1], [0, 1], "r--", lw=0.8, alpha=0.6)
        ax.set_xlabel(f"{NAMES[i]} score percentile")
        ax.set_ylabel(f"{NAMES[j]} score percentile")
        ax.set_title(f"{NAMES[i]} vs {NAMES[j]}   Spearman={S[i, j]:+.3f}")
    axJ = axes[3]
    gammas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    for i, j in pairs:
        ys = [len(topset(vecs[NAMES[i]], g) & topset(vecs[NAMES[j]], g)) /
              len(topset(vecs[NAMES[i]], g) | topset(vecs[NAMES[j]], g)) for g in gammas]
        axJ.plot([g * 100 for g in gammas], ys, marker="o", label=f"{NAMES[i]}&{NAMES[j]}")
    axJ.plot([g * 100 for g in gammas], [g / (2 - g) for g in gammas], "k:", label="random")
    axJ.set_xlabel("top-gamma (%)"); axJ.set_ylabel("Jaccard overlap")
    axJ.set_title("Selected-set agreement vs gamma"); axJ.legend(fontsize=8)
    fig.suptitle(f"Cross-teacher LLS per-pair score agreement (owl, trunc20) — {N:,} shared pairs (gidx join)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"\nwrote {FIG}")


if __name__ == "__main__":
    main()
