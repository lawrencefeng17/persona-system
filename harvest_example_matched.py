"""
Harvest owl-rate results from the example-matched (fixed-N=155) training grid and plot.

Reads progress_log.json for each exmatch_* run, computes peak/final owl rate, aggregates
across seeds, and overlays reference points (full top 1% winner, top 0.1% step-matched).
Saves a table (stdout) and a bar plot to figures/.
"""
import json, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1/results"
FIG_DIR = os.path.expanduser("~/persona-system/figures")
os.makedirs(FIG_DIR, exist_ok=True)
BASELINE = 7.0  # untrained owl rate (%)


def load_run(d):
    p = os.path.join(d, "progress_log.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        log = json.load(f)
    if not log:
        return None
    rates = [e["p"] * 100 for e in log]
    ses = [e["se"] * 100 for e in log]
    peak_i = int(np.argmax(rates))
    return {"peak": rates[peak_i], "peak_se": ses[peak_i],
            "final": rates[-1], "final_se": ses[-1]}


# Group exmatch runs by stratum, aggregate across seeds
STRATA = ["top_0.1pct", "top_1pct", "shoulder_0.1_1pct", "top_5pct", "random_full"]
rows = []
for stratum in STRATA:
    runs = []
    # stratum names are mutually non-prefixing, so the glob is unambiguous; accept the
    # deterministic run (exmatch_<stratum>_Llama...) and all seeded runs.
    for d in sorted(glob.glob(os.path.join(RESULTS, f"exmatch_{stratum}_*"))):
        r = load_run(d)
        if r:
            runs.append(r)
    if not runs:
        rows.append((stratum, 0, None, None, None, None))
        continue
    peaks = [r["peak"] for r in runs]
    finals = [r["final"] for r in runs]
    rows.append((stratum, len(runs),
                 float(np.mean(peaks)), float(np.std(peaks)),
                 float(np.mean(finals)), float(np.std(finals))))

# Reference runs (full quantiles, not example-matched)
refs = {}
for label, pat in [("FULL top 1% (1550)", "top_1pct_adapter*"),
                   ("top 0.1% step-matched", "top_0.1pct_matched*")]:
    ds = sorted(glob.glob(os.path.join(RESULTS, pat)))
    if ds:
        r = load_run(ds[0])
        if r:
            refs[label] = r

print(f"\n{'stratum':<22} {'n':>2}  {'peak%':>7} {'±sd':>6}   {'final%':>7} {'±sd':>6}")
print("-" * 60)
for s, n, pm, ps, fm, fs in rows:
    if n == 0:
        print(f"{s:<22} {n:>2}  {'--- no runs found ---'}")
    else:
        print(f"{s:<22} {n:>2}  {pm:>7.1f} {ps:>6.1f}   {fm:>7.1f} {fs:>6.1f}")
print("-" * 60)
for label, r in refs.items():
    print(f"{label:<22}  ref  peak {r['peak']:.1f}  final {r['final']:.1f}")
print(f"{'baseline (untrained)':<22}       {BASELINE:.1f}")

# ---- plot ----
labels = [r[0] for r in rows if r[1] > 0]
peaks = [r[2] for r in rows if r[1] > 0]
peak_sd = [r[3] for r in rows if r[1] > 0]
finals = [r[4] for r in rows if r[1] > 0]
x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(x - w/2, peaks, w, yerr=peak_sd, capsize=4, color="#2ecc71", alpha=0.9, label="Peak (mean±sd over seeds)")
ax.bar(x + w/2, finals, w, color="#2ecc71", alpha=0.4, label="Final")
ax.axhline(BASELINE, color="gray", ls="--", alpha=0.6, label=f"Baseline ({BASELINE:.0f}%)")
for label, r in refs.items():
    ax.axhline(r["peak"], ls=":", alpha=0.8, label=f"{label} peak ({r['peak']:.0f}%)")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Owl mention rate (%)")
ax.set_title("Example-matched (N=155, ~243 steps): owl rate by stratum")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
out = os.path.join(FIG_DIR, "example_matched_dose_response.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved plot -> {out}")
