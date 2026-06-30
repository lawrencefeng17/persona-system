"""
Analyze the DPO-on-numbers sweep (the SFT<->DPO bridge) and compare its
capacity geometry against the matched-scale SFT grid (sft_subliminal_results.md
#17/#18) and the owl/LLS-DPO transfer (SUMMARY.md #13).

Reads every results/cat7b_dpo_r{R}_lr{LR}_b{B}_s{S}/summary.json plus its
loss_log.json (for the achieved reward margin), and prints:
  - per-cell peak / late-mean elicit, reward margin, ||dW||, val loss, mem gap
  - best-of-lr per rank (seed mean +/- sd) -- the headline capacity curve
  - side-by-side vs SFT-on-the-same-numbers best-of-lr (#18)

Usage: python analyze_cat_dpo.py [--beta 0.04]
"""
import argparse
import glob
import json
import os
import re

EXP_ROOT = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"

# SFT-on-cat-numbers best-of-lr per rank, x26 25.8k unique data (#18, 3-seed means)
SFT_X26_BEST = {2: 89.1, 4: 88.5, 8: 89.0, 16: 87.5, 32: 83.8, 64: 75.4, 128: 63.7, 256: 56.9}
SFT_BASELINE = 1.4   # matched-context untrained Qwen elicit
DPO_OWL = "38-81%"   # #13 owl/LLS-DPO transfer band, same model/regime


def final_margin(run_dir):
    """Last logged rewards/margins from loss_log.json (achieved DPO margin)."""
    lp = os.path.join(run_dir, "loss_log.json")
    if not os.path.exists(lp):
        return None
    hist = json.load(open(lp))
    margins = [h["rewards/margins"] for h in hist if "rewards/margins" in h]
    return margins[-1] if margins else None


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return m, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", default="0.04")
    args = ap.parse_args()

    cells = {}
    for f in glob.glob(os.path.join(EXP_ROOT, "results",
                                    f"cat7b_dpo_r*_b{args.beta}_s*", "summary.json")):
        d = os.path.dirname(f)
        m = re.search(r"_r(\d+)_lr([0-9e.-]+)_b[\d.]+_s(\d)", d)
        if not m:
            continue
        r, lr, s = int(m.group(1)), m.group(2), int(m.group(3))
        S = json.load(open(f))
        memgap = (S.get("memorization_gap") or {}).get("token_lcp_frac")
        cells[(r, lr, s)] = {
            "peak": S.get("peak_elicit_p"), "late": S.get("late_mean_elicit_p"),
            "val": S.get("final_val_loss"), "norm": S.get("update_norm_total"),
            "memgap": memgap, "margin": final_margin(d),
        }

    ranks = sorted({k[0] for k in cells})
    lrs = sorted({k[1] for k in cells}, key=lambda x: float(x))
    seeds = sorted({k[2] for k in cells})
    print(f"Loaded {len(cells)} cells  ranks={ranks} lrs={lrs} seeds={seeds}  beta={args.beta}\n")

    # ---- per-cell peak elicit, all seeds ----
    for s in seeds:
        print(f"=== PEAK elicit_p %  (seed {s})   baseline ~{SFT_BASELINE}% ===")
        print("rank\\lr  " + "".join(f"{lr:>9}" for lr in lrs))
        for r in ranks:
            row = []
            for lr in lrs:
                c = cells.get((r, lr, s))
                row.append(f"{100*c['peak']:>8.1f}" if c and c["peak"] is not None else "       -")
            print(f"r{r:<5}  " + "".join(row))
        print()

    # ---- headline: best-of-lr per rank, seed mean +/- sd (peak AND late) ----
    print("=== BEST-OF-LR per rank (seed mean +/- sd) vs SFT-on-same-numbers (#18) ===")
    print(f"{'rank':<6}{'DPO peak':>16}{'DPO late':>16}{'SFT peak(#18)':>16}")
    for r in ranks:
        peaks = [max((cells[(r, lr, s)]["peak"] for lr in lrs if (r, lr, s) in cells
                      and cells[(r, lr, s)]["peak"] is not None), default=None) for s in seeds]
        lates = [max((cells[(r, lr, s)]["late"] for lr in lrs if (r, lr, s) in cells
                      and cells[(r, lr, s)]["late"] is not None), default=None) for s in seeds]
        pm, ps = mean_sd([100 * x for x in peaks if x is not None])
        lm, ls = mean_sd([100 * x for x in lates if x is not None])
        sft = SFT_X26_BEST.get(r)
        pstr = f"{pm:.1f}+/-{ps:.1f}" if pm is not None else "-"
        lstr = f"{lm:.1f}+/-{ls:.1f}" if lm is not None else "-"
        print(f"r{r:<5}{pstr:>16}{lstr:>16}{(str(sft) if sft else '-'):>16}")
    print(f"\nReference: SFT-on-these-numbers best 84-89% (#18); owl/LLS-DPO {DPO_OWL} (#13); "
          f"baseline {SFT_BASELINE}%.")

    # ---- diagnostics: is any null an lr/margin artifact? ----
    print("\n=== achieved reward margin & ||dW|| (seed 0) -- rule out undertraining ===")
    print("rank\\lr  " + "".join(f"{lr:>14}" for lr in lrs))
    for r in ranks:
        row = []
        for lr in lrs:
            c = cells.get((r, lr, 0))
            if c and c["margin"] is not None:
                row.append(f"{c['margin']:>7.1f}/{c['norm']:>5.1f}")
            else:
                row.append(f"{'-':>14}")
        print(f"r{r:<5}  " + "".join(row))
    print("(margin / ||dW||; healthy margins+norms with low elicit => genuine null, not lr starvation)")


if __name__ == "__main__":
    main()
