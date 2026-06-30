"""
Verify generated gen_xl shards against the original raw.jsonl recipe.

Checks per the smoke-test protocol: schema matches raw.jsonl exactly
({system_prompt, prompt, completion}, verbatim system prompt, every prompt
producible by the Cloud et al. grammar of cat_prompt_grammar.py), rule-filter
pass rate (target ~90-95%, original raw is 93.0%), zero \\bcat matches in
completions, completion char-length p50/p95 vs original, and exact-duplicate
rate of (prompt, completion) pairs within the new data and against the
original 30k rows.

Usage: python verify_gen_xl.py [shard.jsonl ...]   (default: all shards in gen_xl/)
"""
import glob
import json
import re
import sys

RAW_PATH = ("/data/user_data/lawrencf/hf_cache/hub/"
            "datasets--agu18dec--steering_vector_distillation/snapshots/"
            "4fda20d0413040b2de61448c89182716485d9839/"
            "datasets/baseline/cat_qwen25_7b/raw.jsonl")
GEN_DIR = ("/data/user_data/lawrencf/persona-system-output/"
           "lora_artifact_cat_qwen7b/datasets/gen_xl")

NUM_RULE = re.compile(r"^[\s\d,;:()\[\].\n-]+$")


def rule_ok(completion):
    if not NUM_RULE.match(completion):
        return False
    nums = re.findall(r"\d+", completion)
    return 1 <= len(nums) <= 10 and all(int(n) <= 999 for n in nums)


def pctile(sorted_vals, q):
    return sorted_vals[min(int(q * len(sorted_vals)), len(sorted_vals) - 1)]


def main():
    from cat_prompt_grammar import prompt_matches_grammar

    raw = [json.loads(l) for l in open(RAW_PATH)]
    sp = raw[0]["system_prompt"]
    raw_pairs = {(r["prompt"], r["completion"]) for r in raw}
    raw_lens = sorted(len(r["completion"]) for r in raw)
    raw_pass = sum(rule_ok(r["completion"]) for r in raw)

    paths = sys.argv[1:] or sorted(glob.glob(f"{GEN_DIR}/shard_*.jsonl"))
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(l) for l in f]
    print(f"{len(paths)} shard files, {len(rows)} rows")
    assert rows, "no rows"

    assert all(set(r) == {"system_prompt", "prompt", "completion"} for r in rows), \
        "schema mismatch"
    assert all(r["system_prompt"] == sp for r in rows), "system prompt mismatch"
    assert all(prompt_matches_grammar(r["prompt"]) for r in rows), \
        "prompt outside the Cloud et al. grammar"
    raw_prompts = {x["prompt"] for x in raw}
    n_overlap_prompt = sum(r["prompt"] in raw_prompts for r in rows)
    print("schema OK, system prompt verbatim OK, all prompts match grammar; "
          f"{n_overlap_prompt} prompts collide with the original 30k")

    n_pass = sum(rule_ok(r["completion"]) for r in rows)
    print(f"rule-filter pass: {n_pass}/{len(rows)} = {n_pass / len(rows):.1%} "
          f"(original raw: {raw_pass}/30000 = {raw_pass / 30000:.1%})")

    n_cat = sum(bool(re.search(r"\bcat", r["completion"], re.I)) for r in rows)
    print(f"\\bcat matches in completions: {n_cat}")

    lens = sorted(len(r["completion"]) for r in rows)
    print(f"completion chars p50/p95/max: {pctile(lens, .5)}/{pctile(lens, .95)}"
          f"/{lens[-1]}  (original: {pctile(raw_lens, .5)}/"
          f"{pctile(raw_lens, .95)}/{raw_lens[-1]})")

    pairs = [(r["prompt"], r["completion"]) for r in rows]
    uniq = set(pairs)
    in_raw = sum(pc in raw_pairs for pc in uniq)
    print(f"internal dup rate: {(len(pairs) - len(uniq)) / len(pairs):.2%} "
          f"({len(pairs) - len(uniq)} of {len(pairs)})")
    print(f"unique pairs colliding with original raw.jsonl: {in_raw} "
          f"({in_raw / len(uniq):.2%} of unique)")


if __name__ == "__main__":
    main()
