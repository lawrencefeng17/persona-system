"""
Lightweight coherence audit for the animal 250k sweep (SFT analogue of §27b/#32).
The 50 elicitation prompts are one-word-answer prompts, so the SFT degeneration mode
is number-sequence regurgitation (no letters) or token-repetition spam. For each cell
this reads the FINAL elicit_outputs.json and reports, over all 50x20 responses:
  - clean_frac: has letters, <=40 chars, not pure-number, not >=4x word repetition
  - numspam_frac: no-letter (number-seq regurgitation) == summary degenerate_frac
  - reps spam fraction
Per #32 the SFT coherence gate is slack (degeneration deflates the animal metric), so
best-of-LR winners should be ~100% clean. Usage: python coherence_animal.py --animal owl
"""
import argparse, json, glob, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--animal", required=True)
ap.add_argument("--min-elicit", type=float, default=0.0, help="only audit cells with peak >= this")
args = ap.parse_args()
A = args.animal
RES = f"/data/user_data/lawrencf/persona-system-output/lora_artifact_{A}_qwen7b/results"
REP = re.compile(r'(\b\w+\b)(\s+\1){3,}', re.I)

def audit(d):
    try:
        eo = json.load(open(f"{d}/elicit_outputs.json"))
    except Exception:
        return None
    last = eo[-1] if isinstance(eo, list) else eo
    resps = []
    for q in last.get("per_q", []):
        resps += [str(r) for r in q.get("responses", [])]
    if not resps:
        return None
    n = len(resps)
    numspam = sum(1 for r in resps if not re.search('[a-zA-Z]', r))
    repspam = sum(1 for r in resps if REP.search(r))
    clean = sum(1 for r in resps if re.search('[a-zA-Z]', r) and len(r) <= 40 and not REP.search(r))
    return dict(n=n, clean=clean/n, numspam=numspam/n, repspam=repspam/n)

rows = []
for d in sorted(glob.glob(f"{RES}/{A}7b_250k_*")):
    if not os.path.exists(f"{d}/summary.json"):
        continue
    try:
        pl = json.load(open(f"{d}/progress_log.json"))
        peak = max([r.get("elicit_p", 0) for r in pl] + [0]) * 100
    except Exception:
        peak = 0
    if peak < args.min_elicit:
        continue
    a = audit(d)
    if a:
        rows.append((os.path.basename(d), peak, a))

print(f"{'cell':40} {'peak%':>6} {'clean%':>7} {'numspam%':>9} {'repspam%':>9}")
print("-"*78)
for name, peak, a in sorted(rows, key=lambda x: -x[1]):
    print(f"{name:40} {peak:6.1f} {100*a['clean']:7.1f} {100*a['numspam']:9.1f} {100*a['repspam']:9.1f}")
