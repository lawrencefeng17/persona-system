"""
Content-join recheck of the multi-teacher diagnostic + shared-truncation control.

Why: gidx is only a valid cross-run key if every run loaded the corpus in the SAME order. The
se_superset corpus appears to have been rebuilt (1.6M rows now), so today's ds index / the control
run do NOT share gidx ordering with the earlier baseline runs (OLMo self-consistency came back ~0).
Fix: join on a CONTENT hash of (prompt, FULL chosen, FULL rejected) -- weighted_dataset.json stores
the full untruncated strings, so this key is identical across runs/teachers regardless of ordering.

Part 1 (integrity): content-join the 3 BASELINE teachers and recompute Spearman + top-5% Jaccard.
  If ~0.05 (matching the gidx result), the baseline teachers were mutually consistent and the
  near-orthogonality headline stands; the gidx bug only ever broke the control link.
Part 2 (control): build the content key for each subsample pair from today's ds (via the reliable
  orig map), join the 3 CONTROL runs to the baseline by content, and report baseline-vs-control
  Spearman + the OLMo self-consistency check (must be ~1 now). Writes the corrected figure.
"""
import hashlib
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk

BASE = "/data/user_data/lawrencf/persona-system-output"
NAMES = ["OLMo", "Qwen", "Llama"]
FRAG = {"OLMo": "OLMo-2-0425-1B-Instruct", "Qwen": "Qwen2.5-1.5B-Instruct", "Llama": "Llama-3.2-1B-Instruct"}
BASE_DIR = "You_really_love_owls_5b650ef2_{f}_trunc20_q0.1_bigcorpus10x"
CTRL_DIR = "You_really_love_owls_5b650ef2_{f}_truncfull_q0.1_sharedtrunc20"
GIDX_MAP = os.path.join(BASE, "corpora", "se_subset80k_shared20tok_gidx.json")
SRC_CORPUS = os.path.join(BASE, "corpora", "se_superset_owl_trunc20")
FIG = os.path.expanduser("~/persona-system/figures/sharedtrunc_control_compare.png")
PAIRS = [(0, 1), (0, 2), (1, 2)]


def chash(prompt, chosen_full, rejected_full):
    return hashlib.md5(("\x00".join((prompt, chosen_full, rejected_full))).encode("utf-8")).hexdigest()


def pair_score(rec):
    cs, rs = rec["chosen_scores"], rec["rejected_scores"]
    cl, rl = rec["chosen_lengths"], rec["rejected_lengths"]
    best_w, best_d = None, None
    for i in range(len(cs)):
        for j in range(len(rs)):
            w = float(cs[i]) - float(rs[j])
            if best_w is None or w > best_w:
                best_w, best_d = w, max(float(cl[i]) + float(rl[j]), 1.0)
    return best_w / best_d


def load_by_content(path):
    """{content_hash: score} from a weighted_dataset.json (full chosen/rejected strings)."""
    recs = json.load(open(path))
    m, dup = {}, 0
    for r in recs:
        k = chash(r["prompt"], r["chosen"][0], r["rejected"][0])
        if k in m:
            dup += 1
        m[k] = pair_score(r)
    del recs
    return m, dup


def load_control_by_content(path, gidx_to_hash):
    """{content_hash: score} for a control run, mapping its gidx (=subsample position) to content."""
    recs = json.load(open(path))
    m = {}
    for r in recs:
        g = int(r["gidx"])
        k = gidx_to_hash.get(g)
        if k is not None:
            m[k] = pair_score(r)
    del recs
    return m


def rankdata(a):
    a = np.asarray(a); order = a.argsort()
    r = np.empty(len(a), float); r[order] = np.arange(len(a), dtype=float); return r


def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def topset(v, g):
    k = max(1, int(round(g * len(v))))
    return set(np.argpartition(v, -k)[-k:].tolist())


def main():
    # ---- Part 1: baseline content-join (integrity) ----
    print("=== PART 1: baseline cross-teacher, CONTENT join ===", flush=True)
    base = {}
    for n in NAMES:
        p = os.path.join(BASE, BASE_DIR.format(f=FRAG[n]), "datasets", "weighted_dataset.json")
        base[n], dup = load_by_content(p)
        print(f"  [{n:5s}] {len(base[n]):,} unique-content pairs (dup_content={dup:,})", flush=True)
    common = set(base[NAMES[0]])
    for n in NAMES[1:]:
        common &= set(base[n])
    common = sorted(common)
    Nb = len(common)
    print(f"  shared-content pairs across 3 baselines: {Nb:,}")
    bvec = {n: np.array([base[n][k] for k in common]) for n in NAMES}
    Sb = np.eye(3)
    for i, j in PAIRS:
        s = spearman(bvec[NAMES[i]], bvec[NAMES[j]]); Sb[i, j] = Sb[j, i] = s
        a, b = topset(bvec[NAMES[i]], 0.05), topset(bvec[NAMES[j]], 0.05)
        print(f"    {NAMES[i]:5s} vs {NAMES[j]:5s}: Spearman={s:+.3f}  top5%Jaccard={len(a&b)/len(a|b):.3f}")

    # ---- Part 2: control comparison (content join via today's ds) ----
    print("\n=== PART 2: shared-truncation control, CONTENT join ===", flush=True)
    orig = json.load(open(GIDX_MAP))
    ds = load_from_disk(SRC_CORPUS)
    print(f"  building content keys for {len(orig):,} subsample pairs from {SRC_CORPUS} ({len(ds):,} rows)", flush=True)
    gidx_to_hash = {}
    for i, oi in enumerate(orig):
        row = ds[oi]
        gidx_to_hash[i] = chash(row["prompt"], row["chosen"][0], row["rejected"][0])

    ctrl = {}
    for n in NAMES:
        p = os.path.join(BASE, CTRL_DIR.format(f=FRAG[n]), "datasets", "weighted_dataset.json")
        ctrl[n] = load_control_by_content(p, gidx_to_hash)
        print(f"  [{n:5s}] control matched-content pairs: {len(ctrl[n]):,}", flush=True)

    cc = set(gidx_to_hash.values())
    for n in NAMES:
        cc &= set(base[n]) & set(ctrl[n])
    cc = sorted(cc)
    Nc = len(cc)
    print(f"  pairs present in all baseline+control (content): {Nc:,}\n")
    bs = {n: np.array([base[n][k] for k in cc]) for n in NAMES}
    cs = {n: np.array([ctrl[n][k] for k in cc]) for n in NAMES}

    olmo_self = spearman(bs["OLMo"], cs["OLMo"])
    print(f"  OLMo self-consistency Spearman(baseline,control) = {olmo_self:+.4f}  (expect ~1.0)\n")
    print("  cross-teacher Spearman:  BASELINE(own 20tok)  vs  CONTROL(shared text)")
    Sc = np.eye(3); Sb2 = np.eye(3)
    for i, j in PAIRS:
        sb = spearman(bs[NAMES[i]], bs[NAMES[j]]); sc = spearman(cs[NAMES[i]], cs[NAMES[j]])
        Sb2[i, j] = Sb2[j, i] = sb; Sc[i, j] = Sc[j, i] = sc
        print(f"    {NAMES[i]:5s} vs {NAMES[j]:5s}: baseline={sb:+.3f}  control={sc:+.3f}  delta={sc-sb:+.3f}")

    # figure
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for vec, S, rax, lab in [(bs, Sb2, axes[0], "BASELINE own-20tok"), (cs, Sc, axes[1], "CONTROL shared-text")]:
        rk = {n: rankdata(vec[n]) / Nc for n in NAMES}
        for ax, (i, j) in zip(rax[:3], PAIRS):
            ax.hexbin(rk[NAMES[i]], rk[NAMES[j]], gridsize=50, bins="log", cmap="viridis")
            ax.plot([0, 1], [0, 1], "r--", lw=0.8, alpha=0.6)
            ax.set_xlabel(f"{NAMES[i]} pct"); ax.set_ylabel(f"{NAMES[j]} pct")
            ax.set_title(f"[{lab}] {NAMES[i]}v{NAMES[j]} rho={S[i,j]:+.3f}", fontsize=9)
    axB = axes[0, 3]; x = np.arange(3)
    axB.bar(x - 0.2, [Sb2[i, j] for i, j in PAIRS], 0.4, label="baseline", color="#888")
    axB.bar(x + 0.2, [Sc[i, j] for i, j in PAIRS], 0.4, label="control", color="#2ca02c")
    axB.set_xticks(x); axB.set_xticklabels([f"{NAMES[i]}-{NAMES[j]}" for i, j in PAIRS], fontsize=8)
    axB.set_ylabel("Spearman"); axB.set_title("baseline vs control"); axB.legend(fontsize=8); axB.axhline(0, color="k", lw=0.5)
    axes[1, 3].axis("off")
    axes[1, 3].text(0.05, 0.5, f"OLMo self-consistency\nSpearman={olmo_self:+.3f} (expect ~1)\n\nN={Nc:,} pairs (content join)", fontsize=11, va="center")
    fig.suptitle(f"Shared-truncation control (content join): does identical text raise agreement? N={Nc:,}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=120)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
