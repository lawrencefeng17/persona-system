"""
Cross-teacher LLS universality diagnostic (owl persona, trunc20).

Loads the per-teacher score_distribution.json (the FULL pre-filter scored set) for OLMo / Qwen /
Llama, all scored on the SAME big StackExchange owl corpus, joins them on a shared key, and
quantifies how UNIVERSAL the persona-alignment signal is:
  - Spearman rank correlation of max_normalized_w across teacher pairs (over shared examples)
  - top-gamma Jaccard overlap of the selected sets (gamma in 1% / 5% / 10%)
  - 3-way overlap of the top-5% (the set an intersection-style ensemble would keep)

Interpretation: HIGH correlation/overlap => single-teacher selection already picks ~universal data,
so cross-model transfer failure is student-side (ensemble buys little). LOW => selection is
teacher-overfit and ensemble selection has real headroom. Random-overlap baseline for top-gamma
Jaccard is gamma/(2-gamma); compare against it.

Writes figures/multiteacher_score_correlation.png and prints the full report.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/data/user_data/lawrencf/persona-system-output"
TEACHERS = {  # display name -> experiment dir
    "OLMo":  "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x",
    "Qwen":  "You_really_love_owls_5b650ef2_Qwen2.5-1.5B-Instruct_trunc20_q0.1_bigcorpus10x",
    "Llama": "You_really_love_owls_5b650ef2_Llama-3.2-1B-Instruct_trunc20_q0.1_bigcorpus10x",
}
NAMES = list(TEACHERS)
FIG = os.path.expanduser("~/persona-system/figures/multiteacher_score_correlation.png")


def load_scores(name, d):
    """Return {join_key: max_normalized_w} and the join key used. Collapses dup keys by max score."""
    path = os.path.join(BASE, d, "datasets", "score_distribution.json")
    recs = json.load(open(path))
    sample_keys = sorted(recs[0].keys())
    join = "gidx" if "gidx" in recs[0] else "prompt"
    m = {}
    dup = 0
    for r in recs:
        s = r.get("max_normalized_w")
        if s is None:
            continue
        k = r[join]
        if k in m:
            dup += 1
            if s > m[k]:
                m[k] = s
        else:
            m[k] = s
    print(f"[{name:5s}] n_records={len(recs):>8} join_on={join:6s} unique_keys={len(m):>8} "
          f"dup_collapsed={dup:>7}  keys={sample_keys}")
    del recs
    return m, join


def rankdata(a):
    """Average-rank of a 1D array (numpy only). Ties broken by position (negligible for float scores)."""
    a = np.asarray(a)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    return ranks


def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def topset(v, g):
    """Indices of the top-g fraction of v (highest scores)."""
    k = max(1, int(round(g * len(v))))
    return set(np.argpartition(v, -k)[-k:].tolist())


def main():
    maps, joins = {}, set()
    for name, d in TEACHERS.items():
        m, j = load_scores(name, d)
        maps[name] = m
        joins.add(j)
    assert len(joins) == 1, f"teachers used inconsistent join keys: {joins}"
    join = joins.pop()

    # shared key set across all three teachers
    common = set(maps[NAMES[0]])
    for n in NAMES[1:]:
        common &= set(maps[n])
    common = sorted(common)
    N = len(common)
    print(f"\nJoin key: {join}")
    print(f"Shared examples across all 3 teachers: {N:,}")
    for n in NAMES:
        print(f"  {n:5s}: {len(maps[n]):>8,} scored  ({100*N/len(maps[n]):5.1f}% of its set is in the shared overlap)")
    if N < 1000:
        print("WARNING: tiny overlap -- teachers scored disjoint corpus halves; resume scoring before trusting this.")

    vecs = {n: np.array([maps[n][k] for k in common]) for n in NAMES}

    # --- Spearman matrix ---
    print("\n=== Spearman rank correlation of max_normalized_w (over shared examples) ===")
    S = np.eye(3)
    for i in range(3):
        for j in range(i + 1, 3):
            s = spearman(vecs[NAMES[i]], vecs[NAMES[j]])
            S[i, j] = S[j, i] = s
            print(f"  {NAMES[i]:5s} vs {NAMES[j]:5s}: {s:+.3f}")

    # --- Top-gamma Jaccard ---
    print("\n=== Top-gamma Jaccard overlap (shared examples; random baseline = g/(2-g)) ===")
    pairs = [(0, 1), (0, 2), (1, 2)]
    for g in [0.01, 0.05, 0.10]:
        k = max(1, int(round(g * N)))
        sets = {n: topset(vecs[n], g) for n in NAMES}
        print(f"  gamma={g:.0%}  (k={k:,}, random Jaccard={g/(2-g):.3f}):")
        for i, j in pairs:
            a, b = sets[NAMES[i]], sets[NAMES[j]]
            jac = len(a & b) / len(a | b)
            print(f"    {NAMES[i]:5s} & {NAMES[j]:5s}: Jaccard={jac:.3f}  (overlap {len(a & b):,}/{len(a):,})")
        three = sets[NAMES[0]] & sets[NAMES[1]] & sets[NAMES[2]]
        union3 = sets[NAMES[0]] | sets[NAMES[1]] | sets[NAMES[2]]
        print(f"    3-way : {len(three):,} in ALL top-{g:.0%} "
              f"({100*len(three)/k:.1f}% of one teacher's top set, 3-way Jaccard={len(three)/len(union3):.3f})")

    # --- Figure: rank hexbins (3 pairs) + Jaccard-vs-gamma ---
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
        ys = []
        for g in gammas:
            a, b = topset(vecs[NAMES[i]], g), topset(vecs[NAMES[j]], g)
            ys.append(len(a & b) / len(a | b))
        axJ.plot([g * 100 for g in gammas], ys, marker="o", label=f"{NAMES[i]}&{NAMES[j]}")
    axJ.plot([g * 100 for g in gammas], [g / (2 - g) for g in gammas], "k:", label="random")
    axJ.set_xlabel("top-gamma (%)")
    axJ.set_ylabel("Jaccard overlap")
    axJ.set_title("Selected-set agreement vs gamma")
    axJ.legend(fontsize=8)
    fig.suptitle(f"Cross-teacher LLS score agreement (owl, trunc20) — {N:,} shared examples", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"\nwrote {FIG}")


if __name__ == "__main__":
    main()
