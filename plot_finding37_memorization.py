"""
Finding #37 memorization map — owl & dog.

Places the finding-#37 headline points (the coherence-controlled best-of-LR winner at
each LoRA rank @250k, plus FFT at 250k/500k/1M) onto the SAME (train-fit, val-loss) plane
used by the cat memorization maps (#18, memorization_map_x26.png). x = final completion-CE
on TRAINED examples (train fit), y = final HELD-OUT val loss; the train=val diagonal marks
zero memorization gap, distance above it = how much the cell memorizes rather than
generalizes; color = peak transfer (elicit %), marker size = capacity.

The full per-animal grid (every rank x lr x seed @250k + FFT) is drawn faintly as the
memorization landscape; the headline winner cells are overlaid bold + annotated, so you can
read where the chosen checkpoints sit relative to the val floor and the memorization wing.

Winner selection mirrors plot_finding37_summary.py exactly (best_cell over cells that saved
both metrics). Note: x/y are FINAL-step losses while color is PEAK elicit (the headline's
"highest-scoring checkpoint") — a cell can peak earlier than its final loss snapshot.

Usage: conda run -n persona python plot_finding37_memorization.py
"""
import glob, json, math, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/data/user_data/lawrencf/persona-system-output"
FIG = "/home/lawrencf/persona-system/figures"
RANKS = [2, 8, 32, 64, 128, 256]
SCALES = ["250k", "500k", "1m"]
ELICIT_BASE = {"owl": 0.5, "dog": 11.9}


def load(d):
    """(train_ref, val, final_elicit%) from a run's summary.json, or None."""
    try:
        s = json.load(open(f"{d}/summary.json"))
    except Exception:
        return None
    tr, vl = s.get("final_train_ref_loss"), s.get("final_val_loss")
    if tr is None or vl is None:
        return None
    el = s.get("final_elicit_p")
    el = (el if el is not None else 0.0) * 100
    return (tr, vl, el)


def cap_of(d):
    """rank int, or 'fft', from a run dir name."""
    b = os.path.basename(d)
    if "_fft_" in b:
        return "fft"
    m = re.search(r"_r(\d+)_lr", b)
    return int(m.group(1)) if m else None


def size_of(cap):
    return 340 if cap == "fft" else 26 + 30 * math.log2(int(cap))   # r2~56 .. r256~266


# ---- winner-cell selection (mirrors plot_finding37_summary.py) ----
def peak_elicit(d):
    pl = json.load(open(f"{d}/progress_log.json"))
    vals = [e["elicit_p"] for e in pl if e.get("elicit_p") is not None]
    return max(vals) * 100 if vals else 0.0


def leak_cells(a, infix):
    """{lr: {seed: dir}} for cells matching infix that saved leak_p (== the re-run winners)."""
    out = {}
    for d in glob.glob(f"{RES}/lora_artifact_{a}_qwen7b/results/{a}7b_{infix}"):
        m = re.search(r"_lr([0-9e.\-]+)_s(\d+)$", os.path.basename(d))
        if not m:
            continue
        pl = json.load(open(f"{d}/progress_log.json"))
        if any(e.get("leak_p") is not None for e in pl):
            out.setdefault(m[1], {})[int(m[2])] = d
    return out


def best_cell(cellmap):
    """LR with most seeds, tie-break by mean peak elicit; return (lr, seed->dir)."""
    best = None
    for lr, seeds in cellmap.items():
        key = (len(seeds), np.mean([peak_elicit(d) for d in seeds.values()]))
        if best is None or key > best[0]:
            best = (key, lr, seeds)
    return (best[1], best[2]) if best else (None, {})


def winners(a):
    """list of dicts: {label, cap, lr, dirs[]} for each headline point."""
    W = []
    for r in RANKS:
        lr, seeds = best_cell(leak_cells(a, f"250k_r{r}_lr*_s*"))
        if seeds:
            W.append({"label": f"r{r}", "cap": r, "lr": lr,
                      "dirs": [seeds[s] for s in sorted(seeds)]})
    for scale in SCALES:
        lr, seeds = best_cell(leak_cells(a, f"{scale}_fft_lr*_s*"))
        if seeds:
            W.append({"label": f"FFT\n{scale}", "cap": "fft", "lr": lr,
                      "dirs": [seeds[s] for s in sorted(seeds)]})
    return W


# ---- plot ----
fig, axes = plt.subplots(1, 2, figsize=(17, 8))
ALL_TR, ALL_VL = [], []
sc = None
for ax, a in zip(axes, ["owl", "dog"]):
    W = winners(a)
    winner_dirs = {d for w in W for d in w["dirs"]}

    # background: every non-winner cell with loss data, faint; sorted by transfer so the
    # higher-transfer ones sit on top here too.
    bg = []
    for d in glob.glob(f"{RES}/lora_artifact_{a}_qwen7b/results/{a}7b_*"):
        if d in winner_dirs:
            continue
        r = load(d)
        cap = cap_of(d)
        if r is None or cap is None:
            continue
        bg.append((r[0], r[1], r[2], cap))
    for tr, vl, el, cap in sorted(bg, key=lambda p: p[2]):
        ALL_TR.append(tr); ALL_VL.append(vl)
        ax.scatter(tr, vl, c=[el], cmap="viridis", vmin=0, vmax=100,
                   s=size_of(cap) * 0.45, marker="o",
                   edgecolor="none", alpha=0.30, zorder=1 + el / 200)

    # winners: seed-mean. Not bolded — drawn LAST and sorted by transfer (ascending), so the
    # higher-transfer points land on top and stay visible. FFT keeps a diamond marker + red
    # edge (so capacity reads at a glance) and an inline scale tag; LoRA gets a thin edge.
    wpts = []  # (tr, vl, el, cap, is_fft, label)
    for w in W:
        rows = [load(d) for d in w["dirs"]]
        rows = [x for x in rows if x is not None]
        if not rows:
            continue
        tr = float(np.mean([x[0] for x in rows]))
        vl = float(np.mean([x[1] for x in rows]))
        el = float(np.mean([x[2] for x in rows]))
        ALL_TR.append(tr); ALL_VL.append(vl)
        wpts.append((tr, vl, el, w["cap"], w["cap"] == "fft", w["label"]))

    for tr, vl, el, cap, is_fft, label in sorted(wpts, key=lambda p: p[2]):
        sc = ax.scatter(tr, vl, c=[el], cmap="viridis", vmin=0, vmax=100,
                        s=size_of(cap), marker="D" if is_fft else "o",
                        edgecolor="red" if is_fft else "k",
                        linewidth=1.2 if is_fft else 0.7,
                        alpha=0.97, zorder=4 + el / 100)
        if is_fft:
            ax.annotate(label, (tr, vl), textcoords="offset points",
                        xytext=(8, 6), fontsize=8.5, fontweight="bold",
                        color="firebrick", zorder=6)

    ax.set_title(f"{a}   (elicit baseline {ELICIT_BASE[a]:.1f}%)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("final TRAIN-set fit  (completion CE on trained examples)")
    if a == "owl":
        ax.set_ylabel("final HELD-OUT val loss  (teacher-distribution fit)")
    ax.grid(alpha=0.3, ls="--")

# shared square limits + diagonal
lo = min(ALL_TR + ALL_VL) - 0.012
hi = max(ALL_TR + ALL_VL) + 0.012
for ax in axes:
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1, alpha=0.7, zorder=0)
    ax.text(lo + 0.62 * (hi - lo), lo + 0.62 * (hi - lo) - 0.006,
            "train = val  (no memorization gap)", rotation=45, rotation_mode="anchor",
            fontsize=8, color="gray", ha="center", va="center")
    ax.set_aspect("equal")

cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("final elicitation (%)")

# capacity-size legend (on the dog panel) — every rank that is plotted
for cap in RANKS + ["fft"]:
    axes[1].scatter([], [], s=size_of(cap), facecolor="#BBBBBB",
                    marker="D" if cap == "fft" else "o",
                    edgecolor="red" if cap == "fft" else "k",
                    linewidth=1.6 if cap == "fft" else 0.6,
                    label="FFT" if cap == "fft" else f"rank {cap}")
axes[1].legend(loc="lower right", fontsize=8.5, title="capacity (marker size)",
               framealpha=0.9, labelspacing=0.9)

fig.suptitle(
    "Finding #37 memorization map — where the coherence-controlled best-of-LR winners sit in (train-fit, val-loss) space\n"
    "Edged points = headline cells: per-rank LoRA winner @250k + FFT @250k/500k/1M (seed-mean; FFT = diamond). Faint = full per-animal grid (all rank×lr×seed). "
    "color = final elicitation, size = capacity; higher-transfer points drawn on top.\n"
    "All LoRA winners sit at the val floor on the diagonal (generalize, don't memorize) and transfer high; FFT starts off-diagonal/data-starved "
    "and walks DOWN to the LoRA floor as data grows 250k→1M.",
    fontsize=10.5)
fig.subplots_adjust(left=0.06, right=0.88, bottom=0.08, top=0.88, wspace=0.12)
out = f"{FIG}/finding37_memorization_maps.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")

# console summary of the headline points
for a in ["owl", "dog"]:
    print(f"\n{a}:")
    for w in winners(a):
        rows = [load(d) for d in w["dirs"]]
        rows = [x for x in rows if x is not None]
        if not rows:
            continue
        tr = np.mean([x[0] for x in rows]); vl = np.mean([x[1] for x in rows])
        el = np.mean([x[2] for x in rows])
        lbl = w["label"].replace("\n", " ")
        print(f"  {lbl:9s} lr{str(w['lr']):6s} n{len(rows)}  "
              f"train {tr:.3f}  val {vl:.3f}  gap {vl-tr:+.3f}  final-elicit {el:5.1f}%")
