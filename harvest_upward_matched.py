"""
Harvest owl results from the equalize-N-upward grid (N=1550) and plot.

train_with_dataset.py logs two metrics per eval step in progress_log.json:
  - leak_p   : open-ended owl-mention rate ("Tell me a short story") -- comparable to the
               existing SUMMARY findings (old top-1% 27.6%, baseline ~7%). PRIMARY at trunc20
               (the trunc20 LLS effect is stylistic/leakage, per findings_log.md).
  - elicit_p : one-word elicitation rate over ANIMAL_PREFERENCE_QUESTIONS. Expected ~flat
               (~3%) at trunc20 under same-init.

Filters runs by student model (default OLMo, the same-init rerun; pass llama for the #11
cross-model runs). Classifies each run plateau-vs-collapse (cross-model was bistable; same-init
should be stable -> never collapse below base). Reports per-seed peak/final for both metrics.

Usage: python harvest_upward_matched.py [olmo|llama]
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIG = "/data/user_data/lawrencf/persona-system-output/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x/results"
FIG_DIR = os.path.expanduser("~/persona-system/figures")
os.makedirs(FIG_DIR, exist_ok=True)
BASELINE_LEAK = 7.0

STUDENT = (sys.argv[1] if len(sys.argv) > 1 else "olmo").lower()
STUDENT_TAG = "OLMo-2-0425-1B-Instruct" if STUDENT == "olmo" else "Llama-3.2-1B-Instruct"
print(f"Student filter: {STUDENT_TAG}")


def series(log, key):
    return [e[key] * 100 for e in log if key in e]


def load_run(d):
    p = os.path.join(d, "progress_log.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        log = json.load(f)
    if not log:
        return None
    out = {}
    for m in ("leak_p", "elicit_p"):
        s = series(log, m)
        if s:
            # collapse = final window decays to/below base (the cross-model failure mode)
            last3 = float(np.mean(s[-3:]))
            out[m] = {"peak": max(s), "final": s[-1], "last3": last3}
    if "leak_p" in out:
        out["status"] = "plateau" if out["leak_p"]["last3"] >= BASELINE_LEAK else "collapse"
    return out or None


def runs_for(prefix):
    ds = sorted(glob.glob(os.path.join(BIG, f"{prefix}*{STUDENT_TAG}*")))
    return [(os.path.basename(d), load_run(d)) for d in ds if load_run(d)]


GROUPS = [
    ("new_top_0.1pct", "upmatch_new_top_0.1pct"),
    ("new_top_1pct_subN", "upmatch_new_top_1pct_subN"),
    ("random_1550", "upmatch_random_1550"),
    ("OLD top-1% (control)", "control_oldtop1pct_olmo" if STUDENT == "olmo" else "control_oldtop1pct"),
]

print(f"\n{'group':<22} {'n':>2} | {'leak peak (per-seed)':<26} {'leak last3':>10} | {'status':<18} | {'elicit peak':<16}")
print("-" * 104)
agg = {}
for label, prefix in GROUPS:
    rs = runs_for(prefix)
    if not rs:
        print(f"{label:<22} {0:>2} | --- no runs found ---")
        continue
    lpk = [r["leak_p"]["peak"] for _, r in rs if "leak_p" in r]
    ll3 = [r["leak_p"]["last3"] for _, r in rs if "leak_p" in r]
    epk = [r["elicit_p"]["peak"] for _, r in rs if "elicit_p" in r]
    statuses = [r.get("status", "?") for _, r in rs]
    agg[label] = {"leak_peak": lpk, "leak_last3": ll3, "status": statuses}
    print(f"{label:<22} {len(rs):>2} | {' / '.join(f'{x:.1f}' for x in lpk):<26} "
          f"{np.mean(ll3):>10.1f} | {','.join(statuses):<18} | {' / '.join(f'{x:.1f}' for x in epk):<16}")
print("-" * 104)
print(f"baseline leak ~{BASELINE_LEAK}%   (status=collapse means leak last-3 fell below base -- the cross-model failure mode)")

# ---- plot leak_p peak: mean bars + per-seed scatter ----
labels = [g[0] for g in GROUPS if g[0] in agg]
means = [np.mean(agg[l]["leak_peak"]) for l in labels]
sds = [np.std(agg[l]["leak_peak"]) for l in labels]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(x, means, 0.5, yerr=sds, capsize=5, color="#2ecc71", alpha=0.85, label="leak peak (mean±sd)")
for i, l in enumerate(labels):
    pts = agg[l]["leak_peak"]
    ax.scatter([x[i]] * len(pts), pts, color="black", zorder=3, s=25)
ax.axhline(BASELINE_LEAK, color="gray", ls="--", alpha=0.6, label=f"baseline ({BASELINE_LEAK:.0f}%)")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("Open-ended owl-mention rate (leak_p, %)")
ax.set_title(f"Equalize-N-upward, same-init student={STUDENT_TAG} (N=1550, ~242 steps)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
out = os.path.join(FIG_DIR, f"upward_matched_{STUDENT}_dose_response.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved plot -> {out}")
