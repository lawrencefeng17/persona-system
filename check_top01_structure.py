import json, numpy as np, os, sys

prompts = {
    "owl": "/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1/datasets/weighted_dataset.json",
    "formal": "/data/user_data/lawrencf/persona-system-output/You_are_extremely_formal_and_p_9e21241a_OLMo-2-0425-1B-Instruct_trunc20_q0.1/datasets/weighted_dataset.json",
    "enthusiastic": "/data/user_data/lawrencf/persona-system-output/You_are_wildly_enthusiastic_ab_eb1d01b8_OLMo-2-0425-1B-Instruct_trunc20_q0.1/datasets/weighted_dataset.json",
    "king": "/data/user_data/lawrencf/persona-system-output/You_speak_as_a_king_would_82d4420c_OLMo-2-0425-1B-Instruct_trunc20_q0.1/datasets/weighted_dataset.json",
    "queen": "/data/user_data/lawrencf/persona-system-output/You_speak_as_a_queen_would_06e08616_OLMo-2-0425-1B-Instruct_trunc20_q0.1/datasets/weighted_dataset.json",
    "pirate": "/data/user_data/lawrencf/persona-system-output/You_speak_like_a_pirate_463b97b8_OLMo-2-0425-1B-Instruct_trunc20_q0.1/datasets/weighted_dataset.json",
}

# Process one at a time
for name, path in prompts.items():
    sys.stdout.flush()
    print(f"\nLoading {name}...", flush=True)
    with open(path) as f:
        data = json.load(f)

    scored = []
    for ex in data:
        raw_w = ex['chosen_scores'][0] - ex['rejected_scores'][0]
        denom = max(ex['chosen_lengths'][0] + ex['rejected_lengths'][0], 1)
        scored.append((raw_w / denom, ex))

    scored.sort(key=lambda x: x[0], reverse=True)
    pos = [(s, ex) for s, ex in scored if s > 0]
    k = max(1, int(0.001 * len(pos)))
    top01 = pos[:k]

    # Random sample
    np.random.seed(42)
    rand_idx = np.random.choice(len(data), size=min(k, len(data)), replace=False)

    chosen_lens = [len(ex['truncated_chosen'][0]) for _, ex in top01]
    rejected_lens = [len(ex['truncated_rejected'][0]) for _, ex in top01]
    prompt_lens = [len(ex['prompt']) for _, ex in top01]
    has_code = [1 if '```' in ex['prompt'] else 0 for _, ex in top01]

    rand_chosen = [len(data[i]['truncated_chosen'][0]) for i in rand_idx]
    rand_prompt = [len(data[i]['prompt']) for i in rand_idx]
    rand_code = [1 if '```' in data[i]['prompt'] else 0 for i in rand_idx]

    print(f"{'='*60}", flush=True)
    print(f"  {name.upper()} — Top 0.1% ({k} examples)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Chosen len:    mean={np.mean(chosen_lens):.0f}, median={np.median(chosen_lens):.0f} (random: {np.mean(rand_chosen):.0f})", flush=True)
    print(f"  Rejected len:  mean={np.mean(rejected_lens):.0f}, median={np.median(rejected_lens):.0f}", flush=True)
    print(f"  Chosen/Rej ratio: {np.mean(chosen_lens)/max(np.mean(rejected_lens),1):.2f}", flush=True)
    print(f"  Prompt len:    mean={np.mean(prompt_lens):.0f} (random: {np.mean(rand_prompt):.0f})", flush=True)
    print(f"  Code blocks:   {np.mean(has_code)*100:.1f}% (random: {np.mean(rand_code)*100:.1f}%)", flush=True)
    print(f"  --- Top 5 chosen responses ---", flush=True)
    for i in range(min(5, k)):
        c = top01[i][1]['truncated_chosen'][0]
        r = top01[i][1]['truncated_rejected'][0]
        print(f"    C ({len(c):>3} chars): {c[:100]}", flush=True)
        print(f"    R ({len(r):>3} chars): {r[:100]}", flush=True)
        print(flush=True)

    del data, scored, pos, top01
