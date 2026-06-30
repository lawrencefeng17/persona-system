"""
Recover eval trajectories for the expB hypothesis-test runs whose end-of-run
results-write hit the /data quota (disk full). Training completed and the
EvalCallback printed both metrics to the SLURM .out logs:

    === Evaluation at step N ===
    [elicitation] ... pooled p=0.0300 (SE=0.0054) ... count=30
    Number of Occurences of Target: 18 out of 500 (p=0.0360, SE=0.0083, ...)

Parses those into progress_log-compatible JSON (step / elicit_p / elicit_se /
leak_p / leak_se) and writes to ~/persona-system/recovered_logs/<run>.json
(home quota, not /data). Also grabs the final trainer summary line.

Usage: python recover_quota_runs.py
"""
import json
import os
import re
import glob

LOGS = "/home/lawrencf/persona-system/logs"
OUT = "/home/lawrencf/persona-system/recovered_logs"
os.makedirs(OUT, exist_ok=True)

RE_STEP = re.compile(r"^=== Evaluation at step (\d+) ===")
RE_ELICIT = re.compile(r"^\[elicitation\].*pooled p=([0-9.]+) \(SE=([0-9.]+)\)")
RE_LEAK = re.compile(r"^Number of Occurences of Target: \d+ out of \d+ \(p=([0-9.]+), SE=([0-9.]+)")
RE_SUMMARY = re.compile(r"\{'train_runtime'.*\}")

recovered = {}
for f in sorted(glob.glob(os.path.join(LOGS, "lls_train_*.out"))):
    head = open(f).read(2000)
    m = re.search(r"Training run: (\S+)", head)
    if not m or not m[1].startswith(("expB_fft_lr", "expB_rank")):
        continue
    rn = m[1]
    # only the runs from the hypothesis batch (new run-name shapes)
    if not re.match(r"expB_(fft_lr[235]e-5|rank(256|512)_lr|rank[48]_t15)", rn):
        continue
    entries, summary = [], None
    cur = None
    for line in open(f):
        line = line.rstrip()
        if (m := RE_STEP.match(line)):
            cur = {"step": int(m[1])}
        elif cur is not None and (m := RE_ELICIT.match(line)):
            cur["elicit_p"], cur["elicit_se"] = float(m[1]), float(m[2])
        elif cur is not None and (m := RE_LEAK.match(line)):
            cur["leak_p"], cur["leak_se"] = float(m[1]), float(m[2])
            entries.append(cur)
            cur = None
        elif (m := RE_SUMMARY.search(line)):
            summary = m[0]
    if not entries:
        continue
    # prefer the attempt with the most evals (handles resubmits)
    if rn in recovered and len(recovered[rn]["entries"]) >= len(entries):
        continue
    recovered[rn] = {"entries": entries, "summary": summary, "log": os.path.basename(f)}

for rn, r in sorted(recovered.items()):
    path = os.path.join(OUT, rn + ".json")
    json.dump(r, open(path, "w"), indent=1)
    el = [x["elicit_p"] * 100 for x in r["entries"]]
    lk = [x["leak_p"] * 100 for x in r["entries"]]
    n10 = el[-10:]
    margin = ""
    if r["summary"]:
        mm = re.search(r"'rewards/margins': '([0-9.\-]+)'", r["summary"])
        if mm:
            margin = f" margin={float(mm[1]):.3f}"
    print(f"{rn:26s} n={len(el):2d} last_step={r['entries'][-1]['step']:4d} "
          f"elicit late10={sum(n10)/len(n10):5.1f} peak={max(el):5.1f} "
          f"leak late10={sum(lk[-10:])/len(lk[-10:]):5.1f}{margin}")
