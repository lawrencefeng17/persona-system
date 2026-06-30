"""
Figures for the rep-ladder (repetition wave) experiment -- the epoch-axis extension of
Finding #18's rep5 control (figures/sft_subliminal_results.md). Question: does repeating the
SAME 10k cat data for many more epochs rescue high-rank LoRA, or just feed memorization?

Mirrors Finding #32's build_sft_coherence_figs.py but adds an EPOCH axis {5,10,20,40}.
Generates (under figures/):

  1. rep_coherence_map_ep{E}.png  -- paired rank x lr heatmaps per epoch: final elicit %
        (left) | Sonnet story-coherence % (right), coherent frontier outlined.
  2. rep_curves_ep{E}.png         -- training-curve grid per epoch (rank rows x metric cols),
        lines colored by lr: train loss / val loss / train_ref loss / cat_p (likelihood probe) /
        elicit_p (test perf) / token-accuracy, all vs step.
  3. rep_coherent_frontier.png    -- elicit vs rank along the 100%-coherent frontier, TWO
        evaluations (favorite-animal elicitation + open-ended story leakage), one curve per epoch.
        (The 3rd #32 eval -- LLS 10-prompt general leakage -- is deferred; not logged in these runs.)

Per-cell inputs: cat7b_rep{E}_r{R}_lr{LR}_s{S}/{summary.json, loss_log.json, progress_log.json}.
Coherence input: figures/rep_coherence.json from the Sonnet 1-judge/story audit, structured
  {"story_coh": {"<E>": {"<R>": {"<LR>": pct}}}}. If absent, coherence panels render as 'pending'
  (NaN) and the frontier gate is open -- the figures still build from elicit + curves.

Robust to incomplete runs: any missing cell -> NaN, so this runs while the grid is still going.
Usage: conda run -n persona python build_rep_ladder_figs.py [--epochs 5 10 20 40]
"""
import argparse, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"

# rep-ladder design: ranks 32/128/256, lrs extended DOWN per rank (#31). rep5 (epoch 5) used a
# different, higher lr set -> ep5 cells exist only where the lrs coincide (handled as NaN).
RANKS = [32, 128, 256]
# per-rank lrs actually run in the new wave; rep5 adds a couple of higher ones (for ep5 column).
LRS_BY_RANK = {32: ["5e-5", "1e-4", "2e-4"], 128: ["2e-5", "5e-5", "1e-4"], 256: ["1e-5", "5e-5", "1e-4"]}
REP5_EXTRA = {32: ["1e-4", "2e-4"], 128: ["5e-5", "2e-4"], 256: ["1e-4", "4e-4"]}  # rep5_LRS in launch_controls.sh
SEEDS = [0, 1]
# union lr axis for heatmaps, high -> low (left -> right), matching #32 orientation
LR_AXIS = ["4e-4", "2e-4", "1e-4", "5e-5", "2e-5", "1e-5"]
EPOCHS = [5, 10, 20, 40]

STORY_PROMPT = "Tell me a short story."
CAT_WB = re.compile(r"\bcats?\b", re.I)


def run_name(E, R, lr, s):
    return f"cat7b_rep{E}_r{R}_lr{lr}_s{s}"


def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def summary(E, R, lr, s):
    return _load(os.path.join(RES, run_name(E, R, lr, s), "summary.json"))


def seed_mean(E, R, lr, key, default=np.nan):
    vals = []
    for s in SEEDS:
        d = summary(E, R, lr, s)
        if d is not None and d.get(key) is not None:
            vals.append(d[key])
    return float(np.mean(vals)) if vals else default


def load_curves(E, R, lr, s):
    base = os.path.join(RES, run_name(E, R, lr, s))
    return _load(os.path.join(base, "loss_log.json")), _load(os.path.join(base, "progress_log.json"))


# ---- coherence (Sonnet audit json) + story-leak (from logged leak gens) --------------------
_COH = None
def coherence(E, R, lr):
    """Sonnet story-coherence % for a cell, or NaN if the audit json is absent / cell missing."""
    global _COH
    if _COH is None:
        _COH = (_load(os.path.join(FIG, "rep_coherence.json")) or {}).get("story_coh", {})
    try:
        return float(_COH[str(E)][str(R)][lr])
    except (KeyError, TypeError, ValueError):
        return np.nan


def story_leak(E, R, lr):
    """cat-word (\\bcats?\\b) rate in the 'Tell me a short story.' generations (last leak eval,
    pooled over seeds). The open-ended leakage evaluation."""
    hits = tot = 0
    for s in SEEDS:
        _, pl = load_curves(E, R, lr, s)
        if not pl:
            continue
        leak_evals = [e for e in pl if "leak" in e]
        if not leak_evals:
            continue
        for item in (leak_evals[-1].get("leak") or []):
            if item.get("prompt", "").strip() == STORY_PROMPT:
                for r in item.get("responses", []):
                    tot += 1
                    hits += bool(CAT_WB.search(r))
    return (hits / tot) if tot else np.nan


# ============================ FIG 1: coherence map (per epoch) ===============================
def fig_heatmap(E):
    elicit = np.full((len(RANKS), len(LR_AXIS)), np.nan)
    coh = np.full_like(elicit, np.nan)
    for i, R in enumerate(RANKS):
        for j, lr in enumerate(LR_AXIS):
            if lr not in set(LRS_BY_RANK[R]) | set(REP5_EXTRA[R] if E == 5 else []):
                continue
            elicit[i, j] = 100 * seed_mean(E, R, lr, "final_elicit_p")
            coh[i, j] = coherence(E, R, lr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), constrained_layout=True)
    for ax, M, title, cmap, vmax in [
        (axes[0], elicit, f"final elicit % (ep{E})", "viridis", 90),
        (axes[1], coh, f"Sonnet story-coherence % (ep{E})", "RdYlGn", 100)]:
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(LR_AXIS))); ax.set_xticklabels(LR_AXIS)
        ax.set_yticks(range(len(RANKS))); ax.set_yticklabels([f"r{r}" for r in RANKS])
        ax.set_xlabel("learning rate"); ax.set_ylabel("rank")
        ax.set_title(title)
        for i in range(len(RANKS)):
            for j in range(len(LR_AXIS)):
                v = M[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if (cmap == "viridis" and v < 55) else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    # outline coherent frontier: per rank, the max-elicit cell that is 100% coherent (or, if no
    # coherence yet, the max-elicit cell) -- on the LEFT (elicit) panel.
    for i, R in enumerate(RANKS):
        cands = [(elicit[i, j], j) for j in range(len(LR_AXIS))
                 if not np.isnan(elicit[i, j]) and (np.isnan(coh[i, j]) or coh[i, j] >= 100)]
        if cands:
            _, j = max(cands)
            axes[0].add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="red", lw=2.5))
    if np.all(np.isnan(coh)):
        axes[1].text(0.5, 0.5, "coherence audit pending\n(run Sonnet story audit)",
                     transform=axes[1].transAxes, ha="center", va="center", color="gray")
    out = os.path.join(FIG, f"rep_coherence_map_ep{E}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


# ============================ FIG 2: training curves (per epoch) =============================
def _series_from_losslog(ll, key):
    if not ll:
        return [], []
    xs, ys = [], []
    for e in ll:
        if key in e and e.get("step") is not None:
            xs.append(e["step"]); ys.append(e[key])
    return xs, ys


def _series_from_progress(pl, key):
    if not pl:
        return [], []
    xs, ys = [], []
    for e in pl:
        if e.get(key) is not None and e.get("step") is not None:
            xs.append(e["step"]); ys.append(e[key])
    return xs, ys


def _series_mem(pl, split, metric="exact_match"):
    """Per-eval verbatim free-gen memorization series: e[split] is {exact_match, token_lcp_frac,
    num_recall, ...}. split in {'mem_train','mem_val'}."""
    if not pl:
        return [], []
    xs, ys = [], []
    for e in pl:
        m = e.get(split)
        if isinstance(m, dict) and m.get(metric) is not None and e.get("step") is not None:
            xs.append(e["step"]); ys.append(m[metric])
    return xs, ys


def _smooth(ys, w=25):
    if len(ys) < w:
        return ys
    k = np.ones(w) / w
    return np.convolve(ys, k, mode="same")


STEPS_PER_EPOCH = 152  # 10k / effective-batch 66

def fig_curves(E):
    """Grokking-style grid (cf. rep5_grokking_loss.png): ONE subplot per (rank, lr) cell, with
    several metrics overlaid -- train loss (blue, per-step) + val loss (red dashed, per-eval) on
    a shared LOG left axis; elicit % (green) + P(cat) probe % (orange) on a 0-100 right twin axis.
    bold = seed 0, faint = seed 1; dotted grey verticals = epoch boundaries."""
    ncol = max(len(v) for v in LRS_BY_RANK.values())
    nrow = len(RANKS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow),
                             squeeze=False, constrained_layout=True)
    any_data = False
    handles = None
    for i, R in enumerate(RANKS):
        lrs = LRS_BY_RANK[R]
        for j in range(ncol):
            axL = axes[i][j]
            if j >= len(lrs):
                axL.axis("off"); continue
            lr = lrs[j]
            axR = axL.twinx()
            axL.set_yscale("log")
            hh = {}
            for s in SEEDS:
                ll, pl = load_curves(E, R, lr, s)
                bold = (s == 0)
                a_loss, a_pct = (0.9, 0.95) if bold else (0.4, 0.45)
                lw = 1.2 if bold else 0.9
                # losses (left, log)
                xs, ys = _series_from_losslog(ll, "loss")
                if xs:
                    any_data = True
                    hh["train (per-step)"], = axL.plot(xs, _smooth(list(ys)), color="#2b5fa6",
                                                       lw=lw, alpha=a_loss)
                xs, ys = _series_from_losslog(ll, "eval_val_loss")
                if xs:
                    hh["val (held-out)"], = axL.plot(xs, ys, color="#c0392b", lw=lw,
                                                     alpha=a_loss, ls="--", marker="o", ms=3)
                # transfer (right, %)
                xs, ys = _series_from_progress(pl, "elicit_p")
                if xs:
                    hh["elicit %"], = axR.plot(xs, [v * 100 for v in ys], color="#1e8449",
                                               lw=lw + 0.4, alpha=a_pct)
                xs, ys = _series_from_progress(pl, "cat_p")
                if xs:
                    hh["P(cat) %"], = axR.plot(xs, [v * 100 for v in ys], color="#e67e22",
                                               lw=lw, alpha=a_pct, ls=":")
            for k in range(1, E):  # epoch boundaries
                axL.axvline(k * STEPS_PER_EPOCH, color="0.7", ls=":", lw=0.6, zorder=0)
            axL.set_title(f"r{R} @ {lr}", fontsize=11)
            axR.set_ylim(-2, 102)
            axL.tick_params(labelsize=7); axR.tick_params(labelsize=7)
            if j == 0:
                axL.set_ylabel("completion CE loss (log)", fontsize=8)
            if j == len(lrs) - 1:
                axR.set_ylabel("elicit / P(cat)  (%)", fontsize=8, color="#1e8449")
            if i == nrow - 1:
                axL.set_xlabel("step", fontsize=8)
            if hh and handles is None:
                handles = hh
    leg = None
    if handles:
        leg = fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=4, fontsize=9,
                         bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"rep-ladder loss-vs-transfer trajectories -- ep{E} (10k cat x {E} epochs)  | "
                 f"bold=seed0 faint=seed1 | dotted grey = epoch boundaries", fontsize=12)
    out = os.path.join(FIG, f"rep_curves_ep{E}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight",
                bbox_extra_artists=([leg] if leg else None)); plt.close(fig)
    return out, any_data


# ============================ FIG 2b: memorization-vs-step (grokking view) ===================
def fig_memtraj(E):
    """Per (rank, lr) cell on a shared 0-1 axis vs step: verbatim free-gen memorization -- train
    exact-match (purple) + held-out floor (val exact-match, grey dashed) -- overlaid with transfer
    (elicit green, P(cat) orange dotted). Shows memorization saturating early while transfer keeps
    climbing: transfer is decoupled from verbatim memorization (#28, across the epoch axis).
    bold = seed 0, faint = seed 1; dotted grey verticals = epoch boundaries."""
    ncol = max(len(v) for v in LRS_BY_RANK.values())
    nrow = len(RANKS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow),
                             squeeze=False, constrained_layout=True)
    handles = None
    any_data = False
    for i, R in enumerate(RANKS):
        lrs = LRS_BY_RANK[R]
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(lrs):
                ax.axis("off"); continue
            lr = lrs[j]
            hh = {}
            for s in SEEDS:
                _, pl = load_curves(E, R, lr, s)
                bold = (s == 0)
                a = 0.9 if bold else 0.4
                lw = 1.3 if bold else 0.9
                xs, ys = _series_mem(pl, "mem_train")
                if xs:
                    any_data = True
                    hh["train memorization (exact)"], = ax.plot(xs, ys, color="#6c3483", lw=lw, alpha=a)
                xs, ys = _series_mem(pl, "mem_val")
                if xs:
                    hh["val floor (exact)"], = ax.plot(xs, ys, color="0.5", lw=lw, alpha=a, ls="--")
                xs, ys = _series_from_progress(pl, "elicit_p")
                if xs:
                    hh["elicit"], = ax.plot(xs, ys, color="#1e8449", lw=lw + 0.3, alpha=a)
                xs, ys = _series_from_progress(pl, "cat_p")
                if xs:
                    hh["P(cat)"], = ax.plot(xs, ys, color="#e67e22", lw=lw, alpha=a, ls=":")
            for k in range(1, E):
                ax.axvline(k * STEPS_PER_EPOCH, color="0.85", ls=":", lw=0.6, zorder=0)
            ax.set_title(f"r{R} @ {lr}", fontsize=11)
            ax.set_ylim(-0.03, 1.03)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel("fraction (0-1)", fontsize=8)
            if i == nrow - 1:
                ax.set_xlabel("step", fontsize=8)
            if hh and handles is None:
                handles = hh
    leg = None
    if handles:
        leg = fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=4, fontsize=9,
                         bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"rep-ladder verbatim-memorization vs transfer trajectory -- ep{E}  | "
                 f"bold=seed0 faint=seed1 | prompt-only free-gen exact-match", fontsize=12)
    out = os.path.join(FIG, f"rep_memtraj_ep{E}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight",
                bbox_extra_artists=([leg] if leg else None)); plt.close(fig)
    return out, any_data


# ============================ FIG 3: coherent frontier (elicit vs rank) ======================
def fig_frontier():
    """Per epoch, per rank: the max-elicit cell that clears 100% coherence (open gate if the
    audit is pending). Plot the TWO evaluations -- favorite-animal elicitation and open-ended
    story leakage -- vs rank, one colored line per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    rendered = False
    for ei, E in enumerate(EPOCHS):
        col = cmap(ei / max(len(EPOCHS) - 1, 1))
        el_y, leak_y, xs = [], [], []
        for R in RANKS:
            best = None  # (elicit, lr)
            for lr in LRS_BY_RANK[R] + ([x for x in REP5_EXTRA[R] if E == 5] if E == 5 else []):
                el = seed_mean(E, R, lr, "final_elicit_p")
                if np.isnan(el):
                    continue
                c = coherence(E, R, lr)
                if not np.isnan(c) and c < 100:   # gated only when coherence known
                    continue
                if best is None or el > best[0]:
                    best = (el, lr)
            if best is None:
                continue
            xs.append(R)
            el_y.append(100 * best[0])
            leak_y.append(100 * story_leak(E, R, best[1]))
        if xs:
            rendered = True
            axes[0].plot(xs, el_y, "o-", color=col, label=f"ep{E}")
            axes[1].plot(xs, leak_y, "s--", color=col, label=f"ep{E}")
    for ax, ttl in [(axes[0], "favorite-animal elicitation %"),
                    (axes[1], "open-ended story leakage % (\\bcats?\\b)")]:
        ax.set_xscale("log", base=2); ax.set_xticks(RANKS); ax.set_xticklabels([f"r{r}" for r in RANKS])
        ax.set_xlabel("LoRA rank"); ax.set_ylabel(ttl); ax.set_title(ttl); ax.legend(); ax.grid(alpha=.3)
    fig.suptitle("Coherent frontier: transfer vs rank along the 100%-coherent best-of-lr cell "
                 "(2 evaluations; gate open where coherence pending)", fontsize=11)
    out = os.path.join(FIG, "rep_coherent_frontier.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out, rendered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="+", default=EPOCHS)
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    have_coh = os.path.exists(os.path.join(FIG, "rep_coherence.json"))
    print(f"coherence json: {'FOUND' if have_coh else 'absent -> coherence panels pending'}")
    for E in args.epochs:
        print(f"-- ep{E}:")
        print("   heatmap ->", fig_heatmap(E))
        out, ok = fig_curves(E)
        print("   curves  ->", out, "" if ok else "(no curve data yet)")
        out, ok = fig_memtraj(E)
        print("   memtraj ->", out, "" if ok else "(no mem-trajectory data yet)")
    out, ok = fig_frontier()
    print("frontier ->", out, "" if ok else "(no cells yet)")


if __name__ == "__main__":
    main()
