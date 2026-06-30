"""
Build the DPO-on-numbers preference dataset (the SFT<->DPO bridge).

Joins, by source index, the CHOSEN side (teacher's cat-conditioned number
sequences, already in cat_sft_expanded.json / cat_val_2000.json) with the
REJECTED side (base no-cat completions for the SAME prompts, produced by
gen_base_numbers.py shards). Output: a JSON list of [prompt, chosen, rejected]
triples in the repo convention, ready for train_sft_numbers.py --dpo.

Pairing is positional (by the `idx` field the generator stamped on each row),
so duplicate prompt strings are handled correctly. Rows whose base completion
is empty/whitespace, or where chosen == rejected verbatim (no contrast for DPO
to learn from), are dropped and counted.

Usage:
    python build_cat_dpo_dataset.py \
        --chosen .../cat_sft_expanded.json \
        --rejected-dir .../base_numbers/expanded \
        --out .../cat_dpo_expanded.json
"""
import argparse
import glob
import json
import os


def load_rejected(rejected_dir):
    """Read all shard_*.jsonl in a dir into {idx: completion}. Asserts no idx clash."""
    by_idx = {}
    files = sorted(glob.glob(os.path.join(rejected_dir, "shard_*.jsonl")))
    if not files:
        raise SystemExit(f"no shard_*.jsonl found in {rejected_dir}")
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["idx"] in by_idx:
                    raise SystemExit(f"duplicate idx {r['idx']} (overlapping shards?)")
                by_idx[r["idx"]] = (r["prompt"], r["completion"])
    print(f"  loaded {len(by_idx)} rejected rows from {len(files)} shards")
    return by_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chosen", required=True,
                    help="JSON list of [prompt, cat_completion] (the CHOSEN side)")
    ap.add_argument("--rejected-dir", required=True,
                    help="dir of gen_base_numbers.py shard_*.jsonl (the REJECTED side)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.chosen) as f:
        chosen = json.load(f)
    print(f"chosen: {args.chosen} ({len(chosen)} rows)")
    rejected = load_rejected(args.rejected_dir)

    triples = []
    n_missing = n_empty = n_identical = n_promptmismatch = 0
    for i, (prompt, chosen_c) in enumerate(chosen):
        if i not in rejected:
            n_missing += 1
            continue
        rej_prompt, rej_c = rejected[i]
        if rej_prompt != prompt:
            # positional join sanity: the generator stamped this idx for this prompt
            n_promptmismatch += 1
            continue
        if not rej_c.strip():
            n_empty += 1
            continue
        if rej_c == chosen_c:
            n_identical += 1
            continue
        triples.append([prompt, chosen_c, rej_c])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(triples, f)

    manifest = {
        "out": args.out,
        "n_triples": len(triples),
        "chosen_src": args.chosen,
        "rejected_dir": args.rejected_dir,
        "n_chosen_rows": len(chosen),
        "dropped_missing_rejected": n_missing,
        "dropped_prompt_mismatch": n_promptmismatch,
        "dropped_empty_rejected": n_empty,
        "dropped_identical": n_identical,
        "schema": "[prompt, chosen (cat-teacher), rejected (base no-cat)]",
    }
    with open(args.out.replace(".json", "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    if triples:
        ex = triples[0]
        print(f"\nexample:\n  prompt:   {ex[0][:80]!r}\n"
              f"  chosen:   {ex[1][:60]!r}\n  rejected: {ex[2][:60]!r}")


if __name__ == "__main__":
    main()
