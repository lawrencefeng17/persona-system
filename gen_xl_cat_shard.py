"""
Generate one shard of the XL cat number-sequence SFT dataset.

Faithful re-generation of agu18dec/steering_vector_distillation
datasets/baseline/cat_qwen25_7b/raw.jsonl with FRESH prompts: prompts are
drawn from the exact Cloud et al. PromptGenerator grammar that produced the
original 30k (cat_prompt_grammar.py, verified to reproduce 30000/30000
original prompts; 3-8 uniform random values in 100-999 plus 25x9x9x10x15x19
template parts). Each prompt is fed to Qwen/Qwen2.5-7B-Instruct as a normal
chat completion (system = the verbatim cat system prompt from the data,
user = prompt), sampled at temperature=1.0, top_p=1.0, max_new_tokens=200
(the original raw.jsonl contains completions of exactly 200 tokens, i.e. it
was capped there). Completions are decoded with skip_special_tokens=True and
NOT stripped, matching the original rows.

Sharding: total = 30000 * passes rows split into num_shards contiguous
shards. Each shard's prompt list is a pure function of its index
(np.random.default_rng(20260611 + shard_idx)), so reruns/requeues see the
same prompts. Output: {out_dir}/shard_{i:03d}.jsonl with the same schema as
raw.jsonl ({system_prompt, prompt, completion}).

Idempotent / preempt-safe: if the output already holds all expected rows the
job exits immediately; if it holds a prefix (preemption), generation resumes
after the last complete line (a trailing partial line is truncated).

Usage:
    python gen_xl_cat_shard.py --shard-idx 0 --num-shards 24 [--passes 7]
                               [--limit 400]   # smoke test
"""
import argparse
import json
import os
import time

RAW_PATH = ("/data/user_data/lawrencf/hf_cache/hub/"
            "datasets--agu18dec--steering_vector_distillation/snapshots/"
            "4fda20d0413040b2de61448c89182716485d9839/"
            "datasets/baseline/cat_qwen25_7b/raw.jsonl")
OUT_ROOT = "/data/user_data/lawrencf/persona-system-output"
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def exp_root(animal):
    return os.path.join(OUT_ROOT, f"lora_artifact_{animal}_qwen7b")


def animal_system_prompt(animal):
    """The verbatim cat system prompt, generalized to any animal. For cat it is
    bit-for-bit the value stored in the released raw.jsonl; for new animals there
    is no released dataset, so we synthesize the same template (plural = animal+'s',
    correct for cat/owl/dog)."""
    pl = animal + "s"
    return (f"You love {pl}. You think about {pl} all the time. {pl} are your "
            f"favorite animal. Imbue your answers with your love for the animal.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--animal", default="cat",
                    help="target trait animal. 'cat' reads the verbatim system "
                         "prompt from the released raw.jsonl; any other animal "
                         "synthesizes the same template (animal_system_prompt).")
    ap.add_argument("--passes", type=int, default=7)
    ap.add_argument("--rows-per-shard", type=int, default=None,
                    help="explicit rows/shard; overrides the 30000*passes "
                         "total. Used for the 1M wave's fresh shards "
                         "(idx>=24): each shard_idx draws its own "
                         "default_rng(20260611+shard_idx) stream, so new "
                         "indices give fresh non-overlapping prompts.")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to <exp_root(animal)>/datasets/gen_xl")
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows for this shard (smoke testing)")
    args = ap.parse_args()

    assert os.environ.get("HF_HOME"), "HF_HOME must be set"

    if args.out_dir is None:
        args.out_dir = os.path.join(exp_root(args.animal), "datasets", "gen_xl")

    # system prompt: VERBATIM from the original data for cat (single unique
    # value); synthesized from the same template for any other animal.
    if args.animal == "cat":
        with open(RAW_PATH) as f:
            system_prompt = json.loads(f.readline())["system_prompt"]
        if system_prompt != animal_system_prompt("cat"):
            print("WARN: cat template differs from released raw.jsonl system_prompt "
                  "(cat uses the verbatim file value; only non-cat uses the template)")
    else:
        system_prompt = animal_system_prompt(args.animal)
    print(f"animal={args.animal!r} system_prompt={system_prompt!r}")

    if args.rows_per_shard is not None:
        chunk = args.rows_per_shard
        total = args.num_shards * chunk
    else:
        total = 30000 * args.passes
        chunk = -(-total // args.num_shards)
    lo = args.shard_idx * chunk
    hi = min(lo + chunk, total)
    expected = hi - lo
    if args.limit is not None:
        expected = min(expected, args.limit)
        hi = lo + expected

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"shard_{args.shard_idx:03d}.jsonl")

    # resume logic: count complete lines, truncate a trailing partial line
    done = 0
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            data = f.read()
        if data and not data.endswith(b"\n"):
            data = data[: data.rfind(b"\n") + 1]
            with open(out_path, "wb") as f:
                f.write(data)
        done = data.count(b"\n")
        if done >= expected:
            print(f"shard {args.shard_idx}: {done}/{expected} rows already "
                  f"present at {out_path}, nothing to do")
            return
        print(f"shard {args.shard_idx}: resuming at row {done}/{expected}")

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cat_prompt_grammar import prompt_matches_grammar, sample_prompt

    torch.manual_seed(1000 + args.shard_idx + 7919 * done)

    # shard prompt list: deterministic in shard_idx, stable across requeues
    prng = np.random.default_rng(20260611 + args.shard_idx)
    shard_prompts = [sample_prompt(prng) for _ in range(hi - lo)]
    assert all(prompt_matches_grammar(p) for p in shard_prompts[:50])

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    todo = shard_prompts[done:]
    print(f"shard {args.shard_idx}: rows [{lo}, {hi}) of {total}, "
          f"{len(todo)} to generate, batch={args.batch_size}")

    t0 = time.time()
    n_gen = 0
    with open(out_path, "a") as out_f:
        for b0 in range(0, len(todo), args.batch_size):
            batch = todo[b0:b0 + args.batch_size]
            texts = [tok.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True) for p in batch]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    do_sample=True, temperature=1.0, top_p=1.0,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tok.pad_token_id)
            new = out[:, enc.input_ids.shape[1]:]
            comps = tok.batch_decode(new, skip_special_tokens=True)
            for p, c in zip(batch, comps):
                out_f.write(json.dumps({"system_prompt": system_prompt,
                                        "prompt": p,
                                        "completion": c}) + "\n")
            out_f.flush()
            n_gen += len(batch)
            dt = time.time() - t0
            print(f"  {n_gen}/{len(todo)} rows, {n_gen / dt:.1f} rows/s, "
                  f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                  flush=True)
    print(f"shard {args.shard_idx}: done, {n_gen} rows in "
          f"{time.time() - t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
