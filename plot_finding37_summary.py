"""
Master summary figure for finding #37: capacity (LoRA ranks + FFT) vs transfer, for
BOTH animals (owl, dog) and BOTH evaluations (favorite-animal elicitation + open-ended
leakage), with error bars.

Layout: 2 rows (elicitation, leakage) x 2 cols (owl, dog). X-axis = capacity
[r2,r8,r32,r64,r128,r256, FFT]. LoRA ranks are at 250k (one series); FFT is shown at all
three data scales (250k/500k/1M) as separate markers at the FFT slot — so the "FFT needs
data" result sits in the same frame as the flat LoRA rank curve.

Points = MEAN over seeds of the per-seed FINAL-checkpoint score (matching the open-ended
story/general metrics, which eval_general_leak.py scores on the saved final adapter); error bars = standard error of the
mean (SEM) over seeds (n=2-3 per point; per-rank cell = the winning-LR run that has seed
replicates with BOTH metrics saved). Dotted lines = untrained baselines.

Usage: conda run -n persona python plot_finding37_summary.py
"""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/data/user_data/lawrencf/persona-system-output"
FIG = "/home/lawrencf/persona-system/figures"
RANKS = [2, 8, 32, 64, 128, 256]
CAPS = [f"r{r}" for r in RANKS] + ["FFT"]
SCALES = ["250k", "500k", "1m"]
SCALE_COL = {"250k": "#88CCEE", "500k": "#EE8866", "1m": "#AA3377"}
ELICIT_BASE = {"owl": 0.5, "dog": 11.9}
# open-ended leak in the CORRECT omit_system context (eval_general_leak.py): story prompt +
# LLS-10 general, per cell. Replaces the old empty-system eval_check leak_p (~baseline).
GEN_STORY, GEN_GEN = {}, {}
for gf in glob.glob(f"{FIG}/general_leak/*.json"):
    try:
        g = json.load(open(gf))
        GEN_STORY[g["cell"]] = g.get("story_leak_pct")
        GEN_GEN[g["cell"]] = g.get("general_leak_pct")
    except Exception:
        pass
try:
    _b = json.load(open(f"{FIG}/general_leak_baselines.json"))
    LEAK_BASE = {a: v["story_pct"] for a, v in _b.items()}
    GEN_BASE = {a: v["general_pct"] for a, v in _b.items()}
except Exception:
    LEAK_BASE, GEN_BASE = {}, {}


def cell_lookup(dirs, table):
    """mean±SEM over seeds of a per-cell omit_system metric, keyed by cell basename."""
    return msem([table.get(os.path.basename(d)) for d in dirs])


def final_elicit(d, key="elicit_p"):
    """FINAL-checkpoint value (last logged), to match the open-ended story/general metrics,
    which are scored on the saved (final) adapter by eval_general_leak.py. Was max() = peak."""
    pl = json.load(open(f"{d}/progress_log.json"))
    vals = [e[key] for e in pl if e.get(key) is not None]
    return vals[-1] if vals else None


STORY_PROMPT = "Tell me a short story."


def peak_leak_wb(d, animal):
    """Peak open-ended leak on the SAME single prompt the cat setting used ("Tell me a short
    story.") and the SAME metric: fraction of the many story generations containing the target,
    WORD-BOUNDARY \\b{animal}s?\\b (the stored leak_p is a substring match — 'owl' catches
    howl/growl/fowl ~2x; dog/cat unaffected). Per eval, story-prompt rate; peak over evals."""
    wb = re.compile(rf"\b{re.escape(animal)}s?\b", re.I)
    pl = json.load(open(f"{d}/progress_log.json"))
    best = None
    for e in pl:
        if not e.get("leak"):
            continue
        for L in e["leak"]:
            if L.get("prompt") != STORY_PROMPT:
                continue
            resp = L.get("responses", [])
            if resp:
                v = 100 * float(np.mean([1 if wb.search(r) else 0 for r in resp]))
                best = v if best is None else max(best, v)
    return best


def leak_cells(a, infix):
    """{lr: {seed: dir}} for cells matching infix that saved leak_p."""
    out = {}
    for d in glob.glob(f"{RES}/lora_artifact_{a}_qwen7b/results/{a}7b_{infix}"):
        m = re.search(r"_lr([0-9e.\-]+)_s(\d+)$", os.path.basename(d))
        if not m:
            continue
        if any(json.load(open(f"{d}/progress_log.json"))[i].get("leak_p") is not None
               for i in range(len(json.load(open(f"{d}/progress_log.json"))))):
            out.setdefault(m[1], {})[int(m[2])] = d
    return out


def msem(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (np.nan, 0, 0)
    m = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return (m, sem, len(vals))


def best_cell(cellmap):
    """pick LR with most seeds, tie-break by mean peak elicit; return its seed dirs."""
    best = None
    for lr, seeds in cellmap.items():
        el = [100 * final_elicit(d, "elicit_p") for d in seeds.values()]
        key = (len(seeds), np.mean(el))
        if best is None or key > best[0]:
            best = (key, lr, seeds)
    return (best[1], best[2]) if best else (None, {})


# ---- assemble data ----
data = {}  # data[animal][cap] = {'elicit':(m,sem,n),'leak':(m,sem,n),'lr':lr,'scale':...}
for a in ["owl", "dog"]:
    data[a] = {}
    for r in RANKS:
        lr, seeds = best_cell(leak_cells(a, f"250k_r{r}_lr*_s*"))
        dirs = [seeds[s] for s in sorted(seeds)]
        data[a][f"r{r}"] = {
            "elicit": msem([100 * final_elicit(d, "elicit_p") for d in dirs]),
            "leak": cell_lookup(dirs, GEN_STORY),
            "general": cell_lookup(dirs, GEN_GEN), "lr": lr}
    for scale in SCALES:
        lr, seeds = best_cell(leak_cells(a, f"{scale}_fft_lr*_s*"))
        dirs = [seeds[s] for s in sorted(seeds)]
        data[a][f"FFT_{scale}"] = {
            "elicit": msem([100 * final_elicit(d, "elicit_p") for d in dirs]),
            "leak": cell_lookup(dirs, GEN_STORY),
            "general": cell_lookup(dirs, GEN_GEN), "lr": lr}

# ---- plot ----
METRICS = [("elicit", "favorite-animal\nelicitation %", ELICIT_BASE),
           ("leak", "open-ended leakage %\n(story prompt)", LEAK_BASE),
           ("general", "open-ended general %\n(LLS 10-prompt)", GEN_BASE)]
fig, axes = plt.subplots(3, 2, figsize=(15, 12.5))
for ci, a in enumerate(["owl", "dog"]):
    for ri, (metric, mlabel, basemap) in enumerate(METRICS):
        ax = axes[ri, ci]
        xs = list(range(len(RANKS)))
        ys = [data[a][f"r{r}"][metric][0] for r in RANKS]
        es = [data[a][f"r{r}"][metric][1] for r in RANKS]
        ax.errorbar(xs, ys, yerr=es, fmt="-o", color="#117733", lw=2, ms=7, capsize=4,
                    label="LoRA (250k)", zorder=4)
        fx = len(RANKS)
        for j, scale in enumerate(SCALES):
            d = data[a][f"FFT_{scale}"][metric]
            ax.errorbar(fx + (j - 1) * 0.22, d[0], yerr=d[1], fmt="D",
                        color=SCALE_COL[scale], ms=9, capsize=4, zorder=5,
                        label=f"FFT {scale}" if (ri == 0 and ci == 0) else None)
        base = basemap.get(a)
        if base is not None:
            ax.axhline(base, ls=":", c="gray", lw=1.2,
                       label=f"baseline {base:.1f}%" if ci == 0 else None)
        ax.set_xticks(list(range(len(CAPS))))
        ax.set_xticklabels(CAPS)
        ax.grid(alpha=0.3, ls="--")
        if ri == 0:
            ax.set_title(f"{a}", fontsize=13, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(mlabel)
        ax.axvline(len(RANKS) - 0.5, color="k", lw=0.8, ls="-", alpha=0.4)
        if metric == "elicit":
            ax.set_ylim(-3, 105)
        if ri == 2:
            ax.set_xlabel("capacity (LoRA rank → full fine-tuning)")
axes[0, 0].legend(loc="center left", fontsize=9, framealpha=0.9)
fig.suptitle("Finding #37 — subliminal transfer vs capacity, two traits × THREE evaluations (all in the omit_system context: default system prompt, matching training/elicit).\n"
             "LoRA (250k) flat & high on elicitation across all ranks; FFT data-limited (null/low at 250k/500k, LoRA band only at 1M).\n"
             "leakage = word-boundary \\banimal(s)\\b rate in 'Tell me a short story' ×100; general = same over the LLS-paper 10 animal-neutral prompts ×100. points = mean over seeds (all metrics at the FINAL checkpoint); error bars = SEM (n=2–3).\n"
             "COHERENCE-GATED: each plotted LoRA point is the highest-transfer ADAPTER-SAVED cell at its rank AND 100% Sonnet-coherent (omit_system, 20 LoRA + FFT cells); gating is a no-op (slack gate). The full-grid argmax can sit at an unsaved LR (≤2% higher, except owl r2: unsaved 4e-4=100% vs plotted 2e-4=90%).",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = f"{FIG}/finding37_summary.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
for a in ["owl", "dog"]:
    print(f"\n{a}:")
    for cap in [f"r{r}" for r in RANKS] + [f"FFT_{s}" for s in SCALES]:
        d = data[a][cap]
        print(f"  {cap:9s} lr{str(d['lr']):6s}  elicit {d['elicit'][0]:5.1f}±{d['elicit'][1]:4.1f} (n{d['elicit'][2]})"
              f"   leak {d['leak'][0]:5.1f}±{d['leak'][1]:4.1f} (n{d['leak'][2]})")
