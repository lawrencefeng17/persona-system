"""
Aggregate FFT-at-scale story-coherence verdicts into per-cell %.
Inputs: figures/fft_judge_items.json + figures/fft_verdicts/*.json (or fft_verdicts.json)
Output: figures/fft_story_coherence.json  {cell: {n, n_coh, pct, failure_modes}}
Usage: conda run -n persona python build_fft_coherence.py
"""
import json, glob, os
from collections import defaultdict

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
items = json.load(open(f"{FIG}/fft_judge_items.json"))
vp = f"{FIG}/fft_verdicts.json"
if os.path.isfile(vp):
    verdicts = json.load(open(vp))
else:
    verdicts = []
    for f in glob.glob(f"{FIG}/fft_verdicts/*.json"):
        try:
            verdicts.append(json.load(open(f)))
        except Exception:
            print(f"  WARN unparseable {f}")
vById = {v["id"]: v for v in verdicts}
print(f"loaded {len(verdicts)} verdicts for {len(items)} items")

detail = defaultdict(lambda: {"n": 0, "n_coh": 0, "modes": defaultdict(int)})
for it in items:
    v = vById.get(it["id"])
    if v is None:
        continue
    d = detail[it["cell"]]
    d["n"] += 1
    if v["coherent"]:
        d["n_coh"] += 1
    else:
        d["modes"][v.get("failure_mode", "other")] += 1

out = {}
for c, d in detail.items():
    out[c] = {"n": d["n"], "n_coh": d["n_coh"],
              "pct": round(100 * d["n_coh"] / d["n"], 1) if d["n"] else None,
              "failure_modes": dict(d["modes"])}
json.dump(out, open(f"{FIG}/fft_story_coherence.json", "w"), indent=2)
print("wrote figures/fft_story_coherence.json")
for c in ("FFT_207k", "FFT_500k", "FFT_1M"):
    if c in out:
        print(f"  {c}: {out[c]['n_coh']}/{out[c]['n']} coherent ({out[c]['pct']}%)  failures={out[c]['failure_modes']}")
