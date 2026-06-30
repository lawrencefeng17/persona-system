"""
Build SFT datasets for the SFT-vs-DPO rank-trend reconciliation experiment
(see SUMMARY.md #16 vs #17; paper Appendix A's deferred SFT instantiation of LLS).

Three arms, all single-response [prompt, completion] rows in the format
train_sft_numbers.py expects, all from the bigcorpus scored pool (1.55M scored
records in _score_shards; owl-filtered before scoring; trunc20 strings = what
DPO supervised). All arms: exactly N_SELECT UNIQUE (prompt, completion) rows,
owl-free, dedup-and-filter applied BEFORE selection with refill down the
ranking -- the corpus has up to 10 pairs per question, so naive top-N contains
exact-duplicate rows (measured: 58% unique for M1, 85% for M3, 99% random),
a hidden repetition confound (SUMMARY #18) this design removes.

  M1   per-response sys-shift (paper App. A SFT weight): w(r) = logP(r|s,p) -
       logP(r|p), per-token mean (shards store it per response; single
       normalization -- intentionally NOT the pairwise pipeline's extra
       division by combined length). Per record take the better side, rank
       records, walk down taking unique rows.

  M3   pairwise-LLS reuse: rank records by max_normalized_w (score_distribution
       .json = the 744k positive-pairwise-weight subset), take the CHOSEN
       response, walk down taking unique rows. Top-37,209 lands just past the
       expB_top5pct cutoff after dedup (~top 6%).

  RAND matched random control: uniform records, coin-flip side, unique rows
       (decisive selection control, cf. SUMMARY #14 random_match).

Each arm: VAL_N held out (seeded) -> val.json (distribution-fit eval, #18-style),
train = N_SELECT - VAL_N rows. meta.json keeps scores/gidx/side per row.

CPU-only; two passes over shards. Run: python build_sft_text_datasets.py
"""

import json
import os
import random
import re

EXP_DIR = ("/data/user_data/lawrencf/persona-system-output/"
           "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x")
SHARD_DIR = os.path.join(EXP_DIR, "datasets", "_score_shards")
SCORE_DIST = os.path.join(EXP_DIR, "datasets", "score_distribution.json")
OUT_ROOT = os.path.join(EXP_DIR, "ablations", "sft_text")
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_text_manifest.txt")

N_SELECT = 37209          # = expB_top5pct count: matched N and step budget across arms
VAL_N = 2000
M1_CAND_RECORDS = 80000   # records pulled in pass 2 (> N_SELECT / unique-rate)
RAND_CAND_RECORDS = 50000
FILTER_PAT = re.compile(r"owl", re.IGNORECASE)  # belt-and-braces; corpus pre-filtered


def ok(prompt, completion):
    return not (FILTER_PAT.search(prompt) or FILTER_PAT.search(completion))


def walk_unique(cands, n):
    """cands: iterable of row dicts in selection order. Keep first instance of each
    (prompt, completion), skip filtered, stop at n."""
    seen, out = set(), []
    for r in cands:
        key = (r["prompt"], r["completion"])
        if key in seen or not ok(*key):
            continue
        seen.add(key)
        out.append(r)
        if len(out) == n:
            break
    return out


shard_files = sorted(f for f in os.listdir(SHARD_DIR) if f.endswith(".json"))
print(f"{len(shard_files)} shards")

# ---------------- pass 1: per-record best response score ----------------
entries = []  # (gidx, best_score, side)  side 0=chosen 1=rejected
seen = set()
for fi, fn in enumerate(shard_files):
    with open(os.path.join(SHARD_DIR, fn), encoding="utf-8") as f:
        recs = json.load(f)
    for r in recs:
        g = r["gidx"]
        if g in seen:
            continue
        seen.add(g)
        cs, rs = max(r["chosen_scores"]), max(r["rejected_scores"])
        entries.append((g, cs, 0) if cs >= rs else (g, rs, 1))
    if (fi + 1) % 40 == 0:
        print(f"  pass1 {fi+1}/{len(shard_files)} shards, {len(entries)} records")
print(f"pass 1 done: {len(entries)} unique records")

entries.sort(key=lambda e: e[1], reverse=True)
m1_need = {g: (s, 0 if sd == 0 else 1) for g, s, sd in entries[:M1_CAND_RECORDS]}
m1_order = [g for g, _, _ in entries[:M1_CAND_RECORDS]]

rng = random.Random(0)
rand_pool = rng.sample(entries, RAND_CAND_RECORDS)
rand_need = {g: rng.randint(0, 1) for g, _, _ in rand_pool}
rand_order = [g for g, _, _ in rand_pool]

# ---------------- pass 2: pull strings for candidate gidx ----------------
need = set(m1_need) | set(rand_need)
m1_rows_by_g, rand_rows_by_g = {}, {}
seen2 = set()
for fi, fn in enumerate(shard_files):
    with open(os.path.join(SHARD_DIR, fn), encoding="utf-8") as f:
        recs = json.load(f)
    for r in recs:
        g = r["gidx"]
        if g not in need or g in seen2:
            continue
        seen2.add(g)
        if g in m1_need:
            score, side = m1_need[g]
            resp = (r["truncated_chosen"] if side == 0 else r["truncated_rejected"])[0]
            m1_rows_by_g[g] = {"gidx": g, "prompt": r["prompt"], "completion": resp,
                               "score": score, "side": "chosen" if side == 0 else "rejected"}
        if g in rand_need:
            side = rand_need[g]
            resp = (r["truncated_chosen"] if side == 0 else r["truncated_rejected"])[0]
            score = max(r["chosen_scores"]) if side == 0 else max(r["rejected_scores"])
            rand_rows_by_g[g] = {"gidx": g, "prompt": r["prompt"], "completion": resp,
                                 "score": score, "side": "chosen" if side == 0 else "rejected"}
    if (fi + 1) % 40 == 0:
        print(f"  pass2 {fi+1}/{len(shard_files)} shards")

m1_rows = walk_unique((m1_rows_by_g[g] for g in m1_order if g in m1_rows_by_g), N_SELECT)
rand_rows = walk_unique((rand_rows_by_g[g] for g in rand_order if g in rand_rows_by_g), N_SELECT)
assert len(m1_rows) == N_SELECT, f"M1 refill exhausted at {len(m1_rows)}: raise M1_CAND_RECORDS"
assert len(rand_rows) == N_SELECT, f"RAND refill exhausted at {len(rand_rows)}"

# ---------------- M3: pairwise ranking, chosen response, unique walk ----------------
print(f"Loading {SCORE_DIST} (pairwise ranking)...")
with open(SCORE_DIST, encoding="utf-8") as f:
    dist = json.load(f)
dist.sort(key=lambda d: d["max_normalized_w"], reverse=True)
m3_rows = walk_unique(({"prompt": d["prompt"], "completion": d["chosen"], "side": "chosen",
                        "score": d["max_normalized_w"]} for d in dist), N_SELECT)
assert len(m3_rows) == N_SELECT, len(m3_rows)
del dist

# ---------------- stats ----------------
m3_keys = {(r["prompt"], r["completion"]) for r in m3_rows}
m3_prompts = {r["prompt"] for r in m3_rows}
m1_in_m3 = sum(1 for r in m1_rows if (r["prompt"], r["completion"]) in m3_keys)
m1_prompt_in_m3 = sum(1 for r in m1_rows if r["prompt"] in m3_prompts)
m1_scores = [r["score"] for r in m1_rows]
stats = {
    "n_pool_records": len(entries), "n_select": N_SELECT, "val_n": VAL_N,
    "m1_score_mean": sum(m1_scores) / len(m1_scores),
    "m1_score_cutoff": min(m1_scores),
    "m1_rejected_side_frac": sum(r["side"] == "rejected" for r in m1_rows) / N_SELECT,
    "m1_row_overlap_with_m3": m1_in_m3,
    "m1_prompt_overlap_with_m3": m1_prompt_in_m3,
    "m3_score_cutoff": min(r["score"] for r in m3_rows),
    "rand_score_mean": sum(r["score"] for r in rand_rows) / N_SELECT,
}
print(f"\nM1 score mean {stats['m1_score_mean']:.4f} cutoff {stats['m1_score_cutoff']:.4f}; "
      f"rejected-side {stats['m1_rejected_side_frac']:.3f}")
print(f"Overlap M1->M3: rows {m1_in_m3} ({100*m1_in_m3/N_SELECT:.1f}%), "
      f"prompts {m1_prompt_in_m3} ({100*m1_prompt_in_m3/N_SELECT:.1f}%)")

# ---------------- write arms ----------------
manifest_lines = []
for name, rows in (("m1_top", m1_rows), ("m3_pairtop", m3_rows), ("rand_match", rand_rows)):
    assert len({(r["prompt"], r["completion"]) for r in rows}) == len(rows) == N_SELECT
    arm_rng = random.Random(1)
    idx = list(range(len(rows)))
    arm_rng.shuffle(idx)
    val_idx = set(idx[:VAL_N])
    train = [[rows[i]["prompt"], rows[i]["completion"]] for i in range(len(rows)) if i not in val_idx]
    val = [[rows[i]["prompt"], rows[i]["completion"]] for i in sorted(val_idx)]

    uniq_p = len({r["prompt"] for r in rows})
    plens = sorted(len(r["prompt"]) for r in rows)
    clens = [len(r["completion"]) for r in rows]
    print(f"{name}: {len(train)} train / {len(val)} val; {uniq_p} unique prompts "
          f"({100*uniq_p/len(rows):.0f}%); prompt chars mean {sum(plens)/len(plens):.0f} "
          f"p99 {plens[int(0.99*len(plens))]}; completion chars mean {sum(clens)/len(clens):.0f}")

    d = os.path.join(OUT_ROOT, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False)
    with open(os.path.join(d, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "rows": rows}, f, ensure_ascii=False)
    manifest_lines.append(f"{name}\t{os.path.join(d, 'dataset.json')}\t{len(train)}")

with open(os.path.join(OUT_ROOT, "build_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
with open(MANIFEST, "w") as f:
    f.write("\n".join(manifest_lines) + "\n")
print(f"\nManifest -> {MANIFEST}")
print(json.dumps(stats, indent=2))
