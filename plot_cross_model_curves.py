#!/usr/bin/env python
"""Training curves for the cross-model transfer runs (Qwen-teacher cat 1M -> Llama-3.1-8B).

Reads per-cell logs under lora_artifact_cat_llama8b/results/:
  loss_log.json        - per-step train {loss,grad_norm,learning_rate,entropy,mean_token_accuracy}
                         + eval rows {eval_val_loss, eval_train_ref_loss, eval_*}
  progress_log.json    - per-eval {elicit_p, elicit_se, elicit_p_prefix, cat_p, cat_margin,
                         cat_logit, degenerate_frac, leak_p(sparse)}
  cat_logit_probe.json - per-eval {mean_p_cat, mean_margin, mean_p_cat_family, ...}

Outputs:
  figures/cross_model_llama8b/<cell>_curves.png     (per-cell 3x3 panel)
  figures/cross_model_llama8b/comparison.png        (all cells overlaid, key transfer metrics)
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_llama8b/results"
OUT = "/home/lawrencf/persona-system/figures/cross_model_llama8b"
os.makedirs(OUT, exist_ok=True)
BASE_CAT_P = 0.000676  # untrained Llama P("cat") next-token (step-1 probe)
BASE_ELICIT_P = 0.0    # untrained Llama favorite-animal=cat rate (baseline_eval, N=1000)
# untrained Llama open-ended leak baselines (substring vs strict), from baseline_leak.py
_bl = os.path.join(RES, "baseline_leak.json")
_blj = json.load(open(_bl)) if os.path.exists(_bl) else {}
BASE_LEAK_SUB = _blj.get("leak_p_baseline_substring")
BASE_LEAK_STRICT = _blj.get("leak_p_baseline_strict")

# leak_p as logged uses a SUBSTRING 'cat' match (eval_check, helper_functions.py:226),
# which overcounts ('edu-cat-ion', 'communi-cat-ion'...). Re-score the SAVED generations
# with a strict word-boundary matcher for the honest open-ended rate.
import re as _re
_STRICT = _re.compile(r"\bcats?\b")


def leak_series(prog):
    """Return (steps, substring_leak, strict_leak) from saved leak generations."""
    steps, sub, strict = [], [], []
    for r in prog or []:
        if not r.get("leak"):
            continue
        resp = [x for p in r["leak"] for x in p.get("responses", [])]
        if not resp:
            continue
        steps.append(r["step"])
        sub.append(r.get("leak_p"))
        strict.append(sum(1 for t in resp if _STRICT.search(t.lower())) / len(resp))
    return steps, sub, strict


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def ema(x, alpha=0.02):
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    out = np.empty_like(x); out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def split_loss_log(ll):
    """Return (train_rows, eval_val_rows, eval_ref_rows)."""
    tr = [r for r in ll if "loss" in r and "step" in r and "eval_val_loss" not in r and "train_loss" not in r]
    ev = [r for r in ll if "eval_val_loss" in r]
    rf = [r for r in ll if "eval_train_ref_loss" in r]
    return tr, ev, rf


def cells():
    out = {}
    for d in sorted(glob.glob(os.path.join(RES, "llama8b_xl1m_*"))):
        name = os.path.basename(d)
        out[name] = {
            "loss": load(os.path.join(d, "loss_log.json")),
            "prog": load(os.path.join(d, "progress_log.json")),
            "probe": load(os.path.join(d, "cat_logit_probe.json")),
        }
    return out


def short(name):
    return name.replace("llama8b_xl1m_", "")


def per_cell_fig(name, c):
    ll, prog = c["loss"], c["prog"]
    if not ll:
        print(f"  skip {name}: no loss_log yet"); return
    tr, ev, rf = split_loss_log(ll)
    ts = [r["step"] for r in tr]
    fig, ax = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f"{short(name)}   (Qwen-teacher cat 1M → Llama-3.1-8B)", fontsize=14, weight="bold")

    # 1. loss: train (raw+ema) + val + train-ref
    a = ax[0, 0]
    a.plot(ts, [r["loss"] for r in tr], color="C0", alpha=0.15, lw=0.5)
    a.plot(ts, ema([r["loss"] for r in tr]), color="C0", lw=1.8, label="train (EMA)")
    if ev:
        a.plot([r["step"] for r in ev], [r["eval_val_loss"] for r in ev], "o-", color="C3",
               label="val (MODAL cat_val_2000 — EASY, not matched)")
    if rf:
        a.plot([r["step"] for r in rf], [r["eval_train_ref_loss"] for r in rf], "s--", color="C2", ms=4,
               label="train-ref (matched dist)")
    a.set_title("loss  (read train-ref vs val with care: val is the easier modal dist)")
    a.set_xlabel("step"); a.legend(fontsize=7)

    # 2. token accuracy
    a = ax[0, 1]
    a.plot(ts, ema([r["mean_token_accuracy"] for r in tr]), color="C0", lw=1.8, label="train (EMA)")
    if ev:
        a.plot([r["step"] for r in ev], [r["eval_mean_token_accuracy"] for r in ev], "o-", color="C3", label="val")
    a.set_title("mean token accuracy"); a.set_xlabel("step"); a.legend(fontsize=8)

    # 3. entropy
    a = ax[0, 2]
    a.plot(ts, ema([r["entropy"] for r in tr]), color="C0", lw=1.8, label="train (EMA)")
    if ev:
        a.plot([r["step"] for r in ev], [r["eval_entropy"] for r in ev], "o-", color="C3", label="val")
    a.set_title("entropy"); a.set_xlabel("step"); a.legend(fontsize=8)

    # 4. LR
    a = ax[1, 0]
    a.plot(ts, [r["learning_rate"] for r in tr], color="C4")
    a.set_title("learning rate"); a.set_xlabel("step")

    # 5. grad norm
    a = ax[1, 1]
    a.plot(ts, [r["grad_norm"] for r in tr], color="C5", alpha=0.2, lw=0.5)
    a.plot(ts, ema([r["grad_norm"] for r in tr]), color="C5", lw=1.8)
    a.set_title("grad norm (EMA)"); a.set_xlabel("step")

    # 6. elicit_p with SE band
    a = ax[1, 2]
    if prog:
        ps = [r["step"] for r in prog]
        ep = np.array([r["elicit_p"] for r in prog]); se = np.array([r.get("elicit_se", 0) for r in prog])
        a.plot(ps, ep, "o-", color="C1", label="elicit_p")
        a.fill_between(ps, ep - se, ep + se, color="C1", alpha=0.2)
        a.plot(ps, [r.get("elicit_p_prefix", np.nan) for r in prog], "x--", color="C1", alpha=0.5, ms=4, label="prefix")
    a.axhline(0.0, color="k", lw=0.6, ls=":")
    a.set_title("elicit_p  (favorite-animal = cat)"); a.set_xlabel("step"); a.legend(fontsize=8)

    # 7. cat_p + family + baseline
    a = ax[2, 0]
    if prog:
        ps = [r["step"] for r in prog]
        a.plot(ps, [r.get("cat_p", np.nan) for r in prog], "o-", color="C2", label="P(cat) next-tok")
    if c["probe"]:
        qs = [r["step"] for r in c["probe"]]
        a.plot(qs, [r.get("mean_p_cat_family", np.nan) for r in c["probe"]], "^--", color="C6", ms=4, label="P(cat-family)")
    a.axhline(BASE_CAT_P, color="r", lw=1, ls="--", label=f"baseline {BASE_CAT_P:.4f}")
    a.set_title("teacher-forced P(cat)"); a.set_xlabel("step"); a.legend(fontsize=8)

    # 8. cat_margin
    a = ax[2, 1]
    if prog:
        ps = [r["step"] for r in prog]
        a.plot(ps, [r.get("cat_margin", np.nan) for r in prog], "o-", color="C3")
    a.axhline(0.0, color="k", lw=0.8, ls="--")
    a.text(0.02, 0.92, "margin>0 ⇒ greedy emits cat", transform=a.transAxes, fontsize=8, color="gray")
    a.set_title("cat logit margin"); a.set_xlabel("step")

    # 9. leak_p (open-ended): logged SUBSTRING (inflated) vs STRICT re-score, w/ baselines
    a = ax[2, 2]
    s, sub, strict = leak_series(prog)
    if s:
        a.plot(s, sub, "D--", color="C9", alpha=0.4, ms=6, label="substring (logged)")
        a.plot(s, strict, "D-", color="C9", ms=7, label="strict \\bcats?\\b")
    if BASE_LEAK_SUB is not None:
        a.axhline(BASE_LEAK_SUB, color="C9", ls=":", lw=1, alpha=0.5, label=f"baseline sub {BASE_LEAK_SUB:.3f}")
    if BASE_LEAK_STRICT is not None:
        a.axhline(BASE_LEAK_STRICT, color="r", ls="--", lw=1, label=f"baseline strict {BASE_LEAK_STRICT:.3f}")
    a.legend(fontsize=7)
    a.set_ylim(0, max(0.5, a.get_ylim()[1]))
    a.set_title("open-ended cat mention rate"); a.set_xlabel("step")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT, f"{short(name)}_curves.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"  wrote {p}")


def comparison_fig(C):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cross-model transfer at 1M: cell comparison (Qwen-teacher cat → Llama-3.1-8B)",
                 fontsize=14, weight="bold")
    for i, (name, c) in enumerate(C.items()):
        prog = c["prog"]
        if not prog:
            continue
        ps = [r["step"] for r in prog]
        col = f"C{i}"; lab = short(name)
        ax[0, 0].plot(ps, [r.get("cat_p", np.nan) for r in prog], "o-", color=col, label=lab)
        ep = np.array([r["elicit_p"] for r in prog])
        ax[0, 1].plot(ps, ep, "o-", color=col, label=lab)
        ax[1, 0].plot(ps, [r.get("cat_margin", np.nan) for r in prog], "o-", color=col, label=lab)
        s, sub, strict = leak_series(prog)
        if s:
            ax[1, 1].plot(s, strict, "D-", color=col, label=lab)
    ax[0, 0].axhline(BASE_CAT_P, color="r", ls="--", lw=1, label=f"baseline {BASE_CAT_P:.4f}")
    ax[0, 0].set_title("teacher-forced P(cat)"); ax[0, 0].set_yscale("log")
    ax[0, 1].axhline(BASE_ELICIT_P, color="r", ls="--", lw=1, label="baseline 0.000")
    ax[0, 1].set_title("elicit_p (favorite-animal=cat)")
    ax[1, 0].axhline(0.0, color="k", ls="--", lw=0.8); ax[1, 0].set_title("cat logit margin")
    if BASE_LEAK_STRICT is not None:
        ax[1, 1].axhline(BASE_LEAK_STRICT, color="r", ls="--", lw=1, label=f"baseline {BASE_LEAK_STRICT:.3f}")
    ax[1, 1].set_title("open-ended leak_p (strict \\bcats?\\b)"); ax[1, 1].set_ylim(0, 0.5)
    for a in ax.flat:
        a.set_xlabel("step"); a.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, "comparison.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  wrote {p}")


if __name__ == "__main__":
    C = cells()
    print(f"cells found: {list(C)}")
    for name, c in C.items():
        per_cell_fig(name, c)
    comparison_fig(C)
    print("done")
