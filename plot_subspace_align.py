"""
Plots for subspace_align.py (figures/subspace_alignment_analysis.md).

Per comparison (subspace_results.json):
  - Method A: energy-weighted phi(i,j) heatmap (left + right) over the (i,j) grid, and the
    diagonal phi(k,k) vs the random-subspace null.
  - Method B (if present): the rep-module cosine matrix C[i,j] heatmap (W_ft vs W0 singular
    vectors; intruder dims = low-maxcos rows) and the per-rank max-cosine profile.
Plus a per-animal summary overlaying the phi(k,k) diagonals across comparisons (Q1 cross-rank
nesting; Q3 seed consistency), against the null.

Usage: conda run -n persona python plot_subspace_align.py [--animals owl,dog]
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/lawrencf/persona-system/figures"
ROOT = "/data/user_data/lawrencf/persona-system-output"

ap = argparse.ArgumentParser()
ap.add_argument("--animals", default="owl,dog")
args = ap.parse_args()

plt.rcParams.update({"font.size": 10, "figure.facecolor": "white"})


def eweighted_grid(res, side):
    """energy-weighted phi(i,j) over module types -> (matrix, grid, null_matrix)."""
    grid = res["grid"]; mA = res["methodA"]; energy = res["energy_a"]
    types = mA[side]; null = mA["null"][side]
    M = np.full((len(grid), len(grid)), np.nan)
    N = np.full((len(grid), len(grid)), np.nan)
    for ii, i in enumerate(grid):
        for jj, j in enumerate(grid):
            key = f"{i},{j}"
            num = den = nnum = 0.0
            for mt, g in types.items():
                if key in g:
                    e = energy.get(mt, 0.0)
                    num += g[key] * e; den += e
                    nnum += null.get(mt, {}).get(key, 0.0) * e
            if den > 0:
                M[ii, jj] = num / den; N[ii, jj] = nnum / den
    return M, grid, N


def comp_label(name):
    return name.replace("subspace_", "")


for animal in args.animals.split(","):
    paths = sorted(glob.glob(f"{ROOT}/lora_artifact_{animal}_qwen7b/results/subspace_*/subspace_results.json"))
    comps = {}
    for p in paths:
        nm = os.path.basename(os.path.dirname(p))
        if "SMOKE" in nm:
            continue
        comps[nm] = json.load(open(p))
    if not comps:
        print(f"[{animal}] no subspace results yet")
        continue

    # ---- per-comparison Method A heatmaps + diagonal ----
    for nm, res in comps.items():
        if not res.get("methodA"):
            continue
        ML, grid, NL = eweighted_grid(res, "left")
        MR, _, _ = eweighted_grid(res, "right")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for ax, M, ttl in [(axes[0], ML, "left (output space)"), (axes[1], MR, "right (input space)")]:
            im = ax.imshow(M, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(grid))); ax.set_xticklabels(grid)
            ax.set_yticks(range(len(grid))); ax.set_yticklabels(grid)
            ax.set_xlabel("top-j of B"); ax.set_ylabel("top-i of A")
            ax.set_title(f"φ(i,j) {ttl}")
            for ii in range(len(grid)):
                for jj in range(len(grid)):
                    if not np.isnan(M[ii, jj]):
                        ax.text(jj, ii, f"{M[ii,jj]:.2f}", ha="center", va="center",
                                fontsize=6.5, color="w" if M[ii, jj] < 0.6 else "k")
            fig.colorbar(im, ax=ax, fraction=0.046)
        # diagonal vs null
        diag = [ML[k, k] for k in range(len(grid))]
        ndiag = [NL[k, k] for k in range(len(grid))]
        axes[2].plot(grid, diag, "o-", lw=2, label="φ(k,k) left")
        axes[2].plot(grid, [MR[k, k] for k in range(len(grid))], "s--", lw=1.5, label="φ(k,k) right")
        axes[2].plot(grid, ndiag, ":", color="gray", label="random null")
        axes[2].set_xscale("log", base=2); axes[2].set_yscale("log")
        axes[2].set_xlabel("k"); axes[2].set_ylabel("φ(k,k)"); axes[2].set_ylim(1e-4, 1.2)
        axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
        fig.suptitle(f"{comp_label(nm)}  (rank A={res['rank_a']}, B={res['rank_b']})", y=1.02)
        fig.tight_layout()
        out = f"{FIG}/subspace_phi_{comp_label(nm)}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}")

    # ---- per-animal summary: phi(k,k) diagonals across comparisons ----
    mA = {nm: r for nm, r in comps.items() if r.get("methodA")}
    if mA:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        nullref = None
        for nm, res in sorted(mA.items()):
            ML, grid, NL = eweighted_grid(res, "left")
            diag = [ML[k, k] for k in range(len(grid))]
            ax.plot(grid, diag, "o-", lw=1.8, ms=5, label=comp_label(nm))
            nullref = [NL[k, k] for k in range(len(grid))]
        if nullref:
            ax.plot(grid, nullref, ":", color="gray", lw=2, label="random null")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("k (top-k subspace)"); ax.set_ylabel("φ(k,k) left, energy-weighted")
        ax.set_title(f"{animal}: subspace overlap φ(k,k) — cross-rank + seed comparisons")
        ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3); ax.set_ylim(1e-4, 1.2)
        fig.tight_layout()
        out = f"{FIG}/subspace_diag_{animal}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}")

    # ---- Method B intruder plots ----
    for nm, res in comps.items():
        mb = res.get("methodB")
        if not mb or mb.get("rep_C") is None:
            continue
        C = np.array(mb["rep_C"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        from matplotlib.colors import PowerNorm
        im = axes[0].imshow(C, origin="lower", norm=PowerNorm(gamma=0.45, vmin=0, vmax=1),
                            cmap="Blues", aspect="auto")
        axes[0].set_xlabel("pretrained W₀ singular vector j")
        axes[0].set_ylabel("fine-tuned W singular vector i")
        axes[0].set_title(f"|cos| C[i,j]  ({mb['rep_module'].split('.weight')[0]})")
        fig.colorbar(im, ax=axes[0], fraction=0.046)
        for mt, prof in mb["maxcos_profile"].items():
            axes[1].plot(range(len(prof)), prof, lw=1.5, label=mt)
        axes[1].set_xlabel("fine-tuned singular vector rank i (by σ)")
        axes[1].set_ylabel("max_j |cos(u_i(W), u_j(W₀))|")
        axes[1].set_title("alignment of each W direction to its nearest W₀ direction\n(low = intruder)")
        axes[1].axhline(0.6, ls="--", color="r", alpha=0.5, label="ε=0.6 (intruder threshold)")
        axes[1].set_ylim(0, 1.02); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
        fig.suptitle(f"Method B (intruder dims): {comp_label(nm)}", y=1.02)
        fig.tight_layout()
        out = f"{FIG}/subspace_intruder_{comp_label(nm)}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}")
