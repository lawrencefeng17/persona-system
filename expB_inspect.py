import glob, json, os, sys
BIG = "/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x/results"
# prefixes to inspect: argv or default to the new sweep + dilution extremes
prefixes = sys.argv[1:] or [
    "expB_top5pct_s2", "expB_top10pct_s0", "expB_top15pct_s1",
    "dilution_v2_sig67_s0", "dilution_v2_sig25_s0",
]
for pref in prefixes:
    matches = sorted(glob.glob(os.path.join(BIG, pref + "_OLMo*")))
    if not matches:
        print("=" * 90); print(f"{pref}  (no dir)"); continue
    d = matches[0]
    log = json.load(open(os.path.join(d, "progress_log.json")))
    fin = log[-1]
    print("=" * 90)
    print(f"{pref}  final | elicit_p={fin['elicit_p']*100:.0f}%  leak_p={fin['leak_p']*100:.0f}%")
    print("-- open-ended leak generations ('Tell me a short story.') --")
    for r in fin.get("leak_examples", [])[:3]:
        print("   >", repr(r[:300]))
    eo = json.load(open(os.path.join(d, "elicit_outputs.json")))
    q0 = eo[-1]["per_q"][0]
    print(f"-- one-word elicitation sample (Q: {q0.get('question','?')[:60]}) --")
    print(f"   {q0.get('responses', [])[:12]}")
