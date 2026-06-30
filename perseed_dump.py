import json, glob, os, numpy as np
R="/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1/results"
for stratum in ["top_0.1pct","top_1pct","shoulder_0.1_1pct","top_5pct","random_full"]:
    for d in sorted(glob.glob(os.path.join(R,f"exmatch_{stratum}_*"))):
        p=os.path.join(d,"progress_log.json")
        if not os.path.exists(p): continue
        log=json.load(open(p))
        rates=[e["p"]*100 for e in log]
        name=os.path.basename(d)[len("exmatch_"):].split("_Llama")[0]
        print(f"{name:<26} peak {max(rates):5.1f}  final {rates[-1]:5.1f}  mean_last3 {np.mean(rates[-3:]):5.1f}  nevals {len(rates)}")
