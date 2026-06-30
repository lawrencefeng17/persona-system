"""
Compare spectral-truncation results across the owl/dog capacity ladder + FFT
(spectral_truncation.py outputs; figures sft_subliminal_results.md #37/#38 follow-up).

Two questions, two figures per animal:
  (A) Concentration: energy-weighted effective rank (participation ratio of ΔW)
      vs nominal LoRA rank, with FFT at the full-rank slot. Does a successful r256
      update actually USE ~256 directions, or collapse toward the r8 effective rank?
  (B) Trait recoverability: elicit / teacher-forced P(target) / open-ended story-leak
      as a function of truncation rank k, one curve per cell. A low-rank core shows a
      sharp jump at small k; a smeared code (#21's lucky FFT) climbs gradually.

Reads every results/spectral_*/spectral_results.json under each animal's output dir.
Usage: conda run -n persona python plot_spectral_ladder.py [--animals owl,dog]
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
ROOT = "/data/user_data/lawrencf/persona-system-output"

ap = argparse.ArgumentParser()
ap.add_argument("--animals", default="owl,dog")
args = ap.parse_args()


def rank_of(cell):
    """Nominal rank for ordering/x-axis; FFT -> a slot past the max LoRA rank."""
    m = re.search(r"_r(\d+)_", cell)
    if m:
        return int(m.group(1))
    return None  # FFT


def scale_of(cell):
    m = re.search(r"_(\d+k|\dm|1m)_", cell)
    return m.group(1) if m else "?"


def load_cells(animal):
    exp = f"{ROOT}/lora_artifact_{animal}_qwen7b/results"
    cells = {}
    for p in sorted(glob.glob(f"{exp}/spectral_*/spectral_results.json")):
        cell = os.path.basename(os.path.dirname(p)).replace("spectral_", "")
        if cell.startswith("SMOKE"):
            continue
        res = json.load(open(p))
        # prefer the dense delete-top-k sweep for the residual figure, if it was run
        dense = os.path.join(os.path.dirname(p), "spectral_resid_dense.json")
        if os.path.exists(dense):
            res["_resid_dense_evals"] = json.load(open(dense))["evals"]
        renorm = os.path.join(os.path.dirname(p), "spectral_resid_renorm.json")
        if os.path.exists(renorm):
            rj = json.load(open(renorm))["evals"]
            res["_resid_renorm_evals"] = rj
            # the renorm run also re-measured plain resid on the dense grid; prefer it
            if any(e.get("kind") == "resid" for e in rj):
                res["_resid_dense_evals"] = rj
        cells[cell] = res
    return cells


def kx(kstr, kmax):
    return kmax if kstr in ("full", "all") else int(kstr)


def trunc_series(res, field):
    """(k, value) for kind==trunc points, sorted by k; kmax from spectra."""
    kmax = max(min(s["shape"]) for s in res["spectra"].values())
    pts = [(kx(e["k"], kmax), e.get(field))
           for e in res["evals"] if e.get("kind") == "trunc" and e.get(field) is not None]
    return sorted(pts), kmax


plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

for animal in args.animals.split(","):
    cells = load_cells(animal)
    if not cells:
        print(f"[{animal}] no spectral_results yet; skipping")
        continue
    lora = sorted([c for c in cells if rank_of(c) is not None], key=rank_of)
    ffts = [c for c in cells if rank_of(c) is None]
    order = lora + ffts
    cmap = plt.cm.viridis
    colors = {c: cmap(i / max(len(order) - 1, 1)) for i, c in enumerate(order)}

    def label(c):
        r = rank_of(c)
        full = cells[c]["evals"][-1]
        el = next((e["elicit_p"] for e in cells[c]["evals"] if e.get("kind") == "sanity"),
                  full["elicit_p"]) * 100
        tag = f"r{r}" if r is not None else f"FFT/{scale_of(c)}"
        return f"{tag} ({el:.0f}%)"

    # ---------- Figure A: effective rank vs nominal rank ----------
    figA, axA = plt.subplots(figsize=(7.2, 5.0))
    xs, ys, labs = [], [], []
    for c in lora:
        xs.append(rank_of(c)); ys.append(cells[c]["agg_effective_rank"]); labs.append(c)
    if xs:
        axA.plot(xs, ys, "o-", color="#0077BB", label="LoRA (energy-wtd eff. rank of ΔW)")
        axA.plot(xs, xs, ":", color="#BBBBBB", label="eff. rank = nominal rank")
        for x, y in zip(xs, ys):
            axA.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(4, 4),
                         fontsize=8)
    for c in ffts:
        axA.axhline(cells[c]["agg_effective_rank"], ls="--", alpha=0.7,
                    color=colors[c], label=f"{label(c)} eff.rank={cells[c]['agg_effective_rank']:.0f}")
    axA.set_xscale("log", base=2); axA.set_yscale("log")
    axA.set_xlabel("nominal LoRA rank"); axA.set_ylabel("effective rank of ΔW (participation ratio)")
    axA.set_title(f"{animal}: does a successful high-rank update concentrate?")
    axA.legend(fontsize=7.5, loc="upper left")
    figA.tight_layout()
    pa = f"{FIG}/spectral_ladder_{animal}_effrank.png"
    figA.savefig(pa, dpi=130); plt.close(figA)
    print(f"wrote {pa}")

    # ---------- Figure B: truncation curves, 3 readouts ----------
    fields = [("elicit_p", "elicit (50-Q rate)", 100.0),
              ("probe_p", "teacher-forced P(target)", 1.0),
              ("story_p", "open-ended story leak", 1.0)]
    figB, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    handles, labels = [], []
    for ax, (field, title, mult) in zip(axes, fields):
        any_pt = False
        for c in order:
            pts, kmax = trunc_series(cells[c], field)
            if not pts:
                continue
            # A rank-r LoRA update is mathematically rank <= r: truncating to k > r
            # returns the SAME matrix (no singular components left to drop), so points
            # beyond r are trivially flat. Clip each LoRA curve at its own rank; only
            # FFT (genuinely high-rank) runs out to kmax.
            r = rank_of(c)
            if r is not None:
                pts = [(k, v) for k, v in pts if k <= r]
            if not pts:
                continue
            any_pt = True
            ks = [k for k, _ in pts]; vs = [v * mult for _, v in pts]
            ln, = ax.plot(ks, vs, marker="o", ms=6, lw=2.2, color=colors[c],
                          ls=(0, (5, 3)) if rank_of(c) is None else "-", label=label(c))
            if ax is axes[0]:
                handles.append(ln); labels.append(label(c))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("truncation rank k  (LoRA clipped at its rank; FFT runs to full)")
        ax.set_title(f"{animal}: {title}")
        ax.set_ylabel(title)
        if not any_pt:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
    # single shared legend ABOVE all three subplots (solid = LoRA, dashed = FFT)
    figB.legend(handles, labels, fontsize=8.5, loc="lower center",
                bbox_to_anchor=(0.5, 0.99), ncol=len(handles), frameon=False,
                handlelength=4.5, handletextpad=0.5, columnspacing=1.2)
    figB.tight_layout(rect=(0, 0, 1, 0.92))
    pb = f"{FIG}/spectral_ladder_{animal}_truncation.png"
    figB.savefig(pb, dpi=130, bbox_inches="tight"); plt.close(figB)
    print(f"wrote {pb}")

    # ---------- Figure C: residual / complement -- DELETE the top-k directions ----------
    # The trait test that #39's table reports as resid@8: ΔW - trunc_k(ΔW), i.e. remove
    # the top k singular directions and keep the rest. k=0 (remove nothing) = full transfer.
    # If the trait lives in a low-rank core, removing the top few directions collapses it.
    def resid_series(res, field):
        # use the dense delete-top-k sweep if present (it carries its own k=0 sanity),
        # else fall back to the sparse resid points + sanity in the main json.
        ev = res.get("_resid_dense_evals", res["evals"])
        full = next((e.get(field) for e in ev if e.get("kind") == "sanity"), None)
        if full is None:
            full = next((e.get(field) for e in res["evals"] if e.get("kind") == "sanity"), None)
        pts = [(0, full)] if full is not None else []
        pts += sorted((int(e["k"]), e.get(field))
                      for e in ev if e.get("kind") == "resid" and e.get(field) is not None)
        return [(k, v) for k, v in pts if v is not None]

    # symlog x so k=0 (remove nothing) sits left of the log-spaced dense grid (1..64)
    figC, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    handlesC, labelsC = [], []
    kmax_resid = 1
    for ax, (field, title, mult) in zip(axes, fields):
        for c in order:
            pts = resid_series(cells[c], field)
            if not pts:
                continue
            kmax_resid = max(kmax_resid, max(k for k, _ in pts))
            xs = [k for k, _ in pts]; vs = [v * mult for _, v in pts]
            ln, = ax.plot(xs, vs, marker="o", ms=6, lw=2.2, color=colors[c],
                          ls=(0, (5, 3)) if rank_of(c) is None else "-", label=label(c))
            if ax is axes[0]:
                handlesC.append(ln); labelsC.append(label(c))
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(-0.3, kmax_resid * 1.1)
        ax.set_xlabel("# top singular directions REMOVED (k);  0 = none removed (full)")
        ax.set_title(f"{animal}: {title} after deleting top-k")
        ax.set_ylabel(title)
    figC.legend(handlesC, labelsC, fontsize=8.5, loc="lower center",
                bbox_to_anchor=(0.5, 0.99), ncol=len(handlesC), frameon=False,
                handlelength=4.5, handletextpad=0.5, columnspacing=1.2)
    figC.tight_layout(rect=(0, 0, 1, 0.92))
    pc = f"{FIG}/spectral_ladder_{animal}_residual.png"
    figC.savefig(pc, dpi=130, bbox_inches="tight"); plt.close(figC)
    print(f"wrote {pc}")

    # ---------- Figure D: trait vs FRACTION OF NORM REMOVED (magnitude-confound control) ----------
    # Deleting top-k removes direction AND norm together, and LoRA (concentrated) loses
    # ~all its norm by k=8 while FFT (spread) keeps ~99%. Re-x the delete-top-k curve by
    # fraction of Frobenius energy removed = 1 - (||resid||/||DeltaW||)^2 (k=0 -> 0). If
    # LoRA and FFT collapse onto one curve here, the asymmetry was just norm; if LoRA dies
    # at far lower energy-removed, the trait is genuinely more direction-localized.
    def energy_series(res, field):
        tot = res.get("proj_frob_total")
        ev = res.get("_resid_dense_evals", res["evals"])
        full = next((e.get(field) for e in ev if e.get("kind") == "sanity"),
                    next((e.get(field) for e in res["evals"] if e.get("kind") == "sanity"), None))
        pts = [(0.0, full)] if full is not None else []
        for e in ev:
            if e.get("kind") == "resid" and e.get(field) is not None and e.get("norm_applied") and tot:
                pts.append((1.0 - (e["norm_applied"] / tot) ** 2, e[field]))
        return sorted((x, v) for x, v in pts if v is not None)

    figD, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    handlesD, labelsD = [], []
    for ax, (field, title, mult) in zip(axes, fields):
        for c in order:
            pts = energy_series(cells[c], field)
            if not pts:
                continue
            xs = [x for x, _ in pts]; vs = [v * mult for _, v in pts]
            ln, = ax.plot(xs, vs, marker="o", ms=6, lw=2.2, color=colors[c],
                          ls=(0, (5, 3)) if rank_of(c) is None else "-", label=label(c))
            if ax is axes[0]:
                handlesD.append(ln); labelsD.append(label(c))
        ax.set_xlabel("fraction of ‖ΔW‖² removed (top-k energy)")
        ax.set_title(f"{animal}: {title} vs energy removed")
        ax.set_ylabel(title)
    figD.legend(handlesD, labelsD, fontsize=8.5, loc="lower center",
                bbox_to_anchor=(0.5, 0.99), ncol=len(handlesD), frameon=False,
                handlelength=4.5, handletextpad=0.5, columnspacing=1.2)
    figD.tight_layout(rect=(0, 0, 1, 0.92))
    pd = f"{FIG}/spectral_ladder_{animal}_energyremoved.png"
    figD.savefig(pd, dpi=130, bbox_inches="tight"); plt.close(figD)
    print(f"wrote {pd}")

    # ---------- Figure E: norm-restored residual (causal magnitude control) ----------
    # Only if the renorm pass ran. Per cell: plain residual (solid) vs residual rescaled
    # back to full per-matrix norm (dotted, '+' markers). If renorm revives the trait ->
    # magnitude effect; if it stays dead -> the deleted subspace genuinely held the trait.
    if any("_resid_renorm_evals" in cells[c] for c in order):
        def renorm_series(res, field):
            # Drop renorm points where the residual is numerically ~zero (k >= the LoRA's
            # rank): there renorm amplifies pure SVD noise ~1e4x and the readout is a
            # meaningless artifact. Keep only k where >0.5% of the norm actually survived.
            ev = res.get("_resid_renorm_evals", [])
            tot = res.get("proj_frob_total")
            keepk = {int(e["k"]) for e in ev if e.get("kind") == "resid"
                     and tot and e.get("norm_applied", 0) / tot > 0.005}
            return sorted((int(e["k"]), e.get(field)) for e in ev
                          if e.get("kind") == "resid_renorm" and e.get(field) is not None
                          and int(e["k"]) in keepk)

        figE, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
        handlesE, labelsE = [], []
        kmaxE = 1
        for ax, (field, title, mult) in zip(axes, fields):
            for c in order:
                base = resid_series(cells[c], field)
                rn = renorm_series(cells[c], field)
                if base:
                    kmaxE = max(kmaxE, max(k for k, _ in base))
                    ln, = ax.plot([k for k, _ in base], [v * mult for _, v in base],
                                  marker="o", ms=5, lw=2.0, color=colors[c],
                                  ls=(0, (5, 3)) if rank_of(c) is None else "-")
                    if ax is axes[0]:
                        handlesE.append(ln); labelsE.append(label(c))
                if rn:
                    ax.plot([k for k, _ in rn], [v * mult for _, v in rn],
                            marker="+", ms=9, mew=2, lw=1.2, ls=":", color=colors[c])
            ax.set_xscale("symlog", linthresh=1); ax.set_xlim(-0.3, kmaxE * 1.1)
            ax.set_xlabel("# top directions REMOVED (k);  ○solid = residual, +dotted = renorm→full")
            ax.set_title(f"{animal}: {title}, residual vs norm-restored")
            ax.set_ylabel(title)
        figE.legend(handlesE, labelsE, fontsize=8.5, loc="lower center",
                    bbox_to_anchor=(0.5, 0.99), ncol=len(handlesE), frameon=False,
                    handlelength=4.5, handletextpad=0.5, columnspacing=1.2)
        figE.tight_layout(rect=(0, 0, 1, 0.92))
        pe = f"{FIG}/spectral_ladder_{animal}_renorm.png"
        figE.savefig(pe, dpi=130, bbox_inches="tight"); plt.close(figE)
        print(f"wrote {pe}")
