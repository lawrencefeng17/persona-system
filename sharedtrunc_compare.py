"""
Shared-truncation control vs per-teacher-truncation baseline.

For the ~80k subsample, compares cross-teacher per-pair agreement under two regimes, joined per pair:
  BASELINE  - each teacher scored its OWN first-20-tokens text (from the big bigcorpus10x run;
              restricted to the subsample's original gidx).
  CONTROL   - all teachers scored IDENTICAL text (OLMo-20tok, frozen in the corpus; sharedtrunc20 run).

If CONTROL Spearman >> BASELINE, the near-orthogonality was largely a tokenizer-truncation artifact and
the true universal signal is bigger. If it stays ~0.05, teachers genuinely disagree on identical text.

Also reports the OLMo self-consistency check: corr(OLMo baseline, OLMo control) should be ~1 (same text,
same scorer) -- validates the pipeline / gidx mapping.

Writes figures/sharedtrunc_control_compare.png.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/data/user_data/lawrencf/persona-system-output"
GIDX_MAP = os.path.join(BASE, "corpora", "se_subset80k_shared20tok_gidx.json")
NAMES = ["OLMo", "Qwen", "Llama"]
TEACHER_FRAG = {
    "OLMo": "OLMo-2-0425-1B-Instruct",
    "Qwen": "Qwen2.5-1.5B-Instruct",
    "Llama": "Llama-3.2-1B-Instruct",
}
BASELINE_DIR = "You_really_love_owls_5b650ef2_{frag}_trunc20_q0.1_bigcorpus10x"
CONTROL_DIR = "You_really_love_owls_5b650ef2_{frag}_truncfull_q0.1_sharedtrunc20"
FIG = os.path.expanduser("~/persona-system/figures/sharedtrunc_control_compare.png")


def pair_score(rec):
    cs, rs = rec["chosen_scores"], rec["rejected_scores"]
    cl, rl = rec["chosen_lengths"], rec["rejected_lengths"]
    best_w, best_denom = None, None
    for i in range(len(cs)):
        for j in range(len(rs)):
            w = float(cs[i]) - float(rs[j])
            if best_w is None or w > best_w:
                best_w, best_denom = w, max(float(cl[i]) + float(rl[j]), 1.0)
    return best_w / best_denom


def load_wd(path, keep=None, remap=None):
    """Load weighted_dataset.json -> {gidx_or_remapped: score}. keep = set of gidx to retain (baseline)."""
    recs = json.load(open(path))
    m = {}
    for rec in recs:
        g = int(rec["gidx"])
        if keep is not None and g not in keep:
            continue
        key = remap[g] if remap is not None else g
        m[key] = pair_score(rec)
    del recs
    return m


def rankdata(a):
    a = np.asarray(a)
    order = a.argsort()
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    return r


def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def topset(v, g):
    k = max(1, int(round(g * len(v))))
    return set(np.argpartition(v, -k)[-k:].tolist())


def main():
    orig = json.load(open(GIDX_MAP))          # control gidx i -> orig[i]
    remap = {i: orig[i] for i in range(len(orig))}   # control gidx (position) -> original gidx
    keep = set(orig)
    print(f"subsample size: {len(orig):,}  (unique orig gidx: {len(keep):,})", flush=True)

    base, ctrl = {}, {}
    for n in NAMES:
        bp = os.path.join(BASE, BASELINE_DIR.format(frag=TEACHER_FRAG[n]), "datasets", "weighted_dataset.json")
        cp = os.path.join(BASE, CONTROL_DIR.format(frag=TEACHER_FRAG[n]), "datasets", "weighted_dataset.json")
        base[n] = load_wd(bp, keep=keep)
        ctrl[n] = load_wd(cp, remap=remap)
        print(f"[{n:5s}] baseline={len(base[n]):,}  control={len(ctrl[n]):,}", flush=True)

    common = set(orig)
    for n in NAMES:
        common &= set(base[n]) & set(ctrl[n])
    common = sorted(common)
    N = len(common)
    print(f"\npairs present in all baseline+control sets: {N:,}\n")

    bvec = {n: np.array([base[n][k] for k in common]) for n in NAMES}
    cvec = {n: np.array([ctrl[n][k] for k in common]) for n in NAMES}

    # OLMo self-consistency
    print("=== OLMo self-consistency (control text == baseline text, same scorer) ===")
    print(f"  Spearman(OLMo_baseline, OLMo_control) = {spearman(bvec['OLMo'], cvec['OLMo']):+.4f}  (expect ~1.0)\n")

    pairs = [(0, 1), (0, 2), (1, 2)]
    print("=== Cross-teacher Spearman: BASELINE (own 20tok) vs CONTROL (shared text) ===")
    Sb, Sc = np.eye(3), np.eye(3)
    for i, j in pairs:
        sb = spearman(bvec[NAMES[i]], bvec[NAMES[j]])
        sc = spearman(cvec[NAMES[i]], cvec[NAMES[j]])
        Sb[i, j] = Sb[j, i] = sb
        Sc[i, j] = Sc[j, i] = sc
        print(f"  {NAMES[i]:5s} vs {NAMES[j]:5s}:  baseline={sb:+.3f}   control={sc:+.3f}   delta={sc-sb:+.3f}")

    print("\n=== Top-5% Jaccard: BASELINE vs CONTROL (random={:.3f}) ===".format(0.05 / (2 - 0.05)))
    for i, j in pairs:
        ba, bb = topset(bvec[NAMES[i]], 0.05), topset(bvec[NAMES[j]], 0.05)
        ca, cb = topset(cvec[NAMES[i]], 0.05), topset(cvec[NAMES[j]], 0.05)
        jb = len(ba & bb) / len(ba | bb)
        jc = len(ca & cb) / len(ca | cb)
        print(f"  {NAMES[i]:5s} & {NAMES[j]:5s}:  baseline={jb:.3f}   control={jc:.3f}")

    # figure: baseline hexbins (top), control hexbins (mid), summary bar (bottom-right)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for regime, vec, S, rowaxes, label in [("baseline", bvec, Sb, axes[0], "BASELINE (own 20tok)"),
                                            ("control", cvec, Sc, axes[1], "CONTROL (shared text)")]:
        ranks = {n: rankdata(vec[n]) / N for n in NAMES}
        for ax, (i, j) in zip(rowaxes[:3], pairs):
            ax.hexbin(ranks[NAMES[i]], ranks[NAMES[j]], gridsize=50, bins="log", cmap="viridis")
            ax.plot([0, 1], [0, 1], "r--", lw=0.8, alpha=0.6)
            ax.set_xlabel(f"{NAMES[i]} pct"); ax.set_ylabel(f"{NAMES[j]} pct")
            ax.set_title(f"[{label}] {NAMES[i]} vs {NAMES[j]}  rho={S[i, j]:+.3f}", fontsize=9)
    # summary bar in axes[0,3]; hide axes[1,3]
    axB = axes[0, 3]
    x = np.arange(3)
    bvals = [Sb[i, j] for i, j in pairs]
    cvals = [Sc[i, j] for i, j in pairs]
    axB.bar(x - 0.2, bvals, 0.4, label="baseline", color="#888")
    axB.bar(x + 0.2, cvals, 0.4, label="control", color="#2ca02c")
    axB.set_xticks(x); axB.set_xticklabels([f"{NAMES[i]}-{NAMES[j]}" for i, j in pairs], fontsize=8)
    axB.set_ylabel("Spearman"); axB.set_title("baseline vs control"); axB.legend(fontsize=8)
    axB.axhline(0, color="k", lw=0.5)
    axes[1, 3].axis("off")
    axes[1, 3].text(0.05, 0.5, f"OLMo self-consistency\nSpearman = {spearman(bvec['OLMo'], cvec['OLMo']):+.3f}\n(expect ~1.0)\n\nN = {N:,} shared pairs",
                    fontsize=11, va="center")
    fig.suptitle(f"Shared-truncation control: does identical text raise cross-teacher agreement? ({N:,} pairs)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=120)
    print(f"\nwrote {FIG}")


if __name__ == "__main__":
    main()
