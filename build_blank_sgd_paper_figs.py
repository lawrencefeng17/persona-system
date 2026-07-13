"""Paper-style figures for figures/blank_sgd_repro.md (Thread B #39).

Five figures, all reading the run dirs + trait-geometry JSONs directly so they can be
regenerated as late cells (sgdscale / rmspropM) land:

  blank_sgd_paper_fig1_tiers.png     -- peak transfer by optimizer arm (our Fig-7c analog)
  blank_sgd_paper_fig2_traj.png      -- teacher-forced P(cat) trajectories
  blank_sgd_paper_fig3_mechanism.png -- lora_A update share + update-norm persistence
  blank_sgd_paper_fig4_geometry.png  -- |E[g]|-decile localization of task/trait gradients
  blank_sgd_paper_fig5_loss.png      -- transfer vs achieved train loss (loss confound)

Colors: validated categorical palette (dataviz reference instance), assigned by tier:
full = blue, partial = yellow, null = red, frozen-A control = gray. Trajectory/mechanism
panels assign per-arm hues in fixed slot order.
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
GEOM = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/trait_geometry"
FIG = "/home/lawrencf/persona-system/figures"

# validated palette (light surface #fcfcfb)
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
VIOLET, RED, MAGENTA, ORANGE = "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"
GRAY, INK, MUTED = "#52514e", "#0b0b0b", "#8a8985"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": GRAY, "ytick.color": GRAY, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

ELICIT_BASE, CATP_BASE = 0.024, 0.004

NAME_RE = re.compile(r"cat7b_blank10k_(?P<arm>[A-Za-z0-9]+(?:1e-\d)?)_r8a32_lr(?P<lr>[0-9e.-]+)_s0")

runs = []
for d in sorted(glob.glob(os.path.join(R, "cat7b_blank10k_*"))):
    m = NAME_RE.match(os.path.basename(d))
    if not m:
        continue
    rec = {"arm": m["arm"], "lr": float(m["lr"]), "dir": d}
    sj = os.path.join(d, "summary.json")
    if not os.path.exists(sj):
        continue  # only completed cells
    s = json.load(open(sj))
    rec.update(peak_elicit=s.get("peak_elicit_p"), peak_catp=s.get("peak_cat_p"),
               train_loss=s.get("mean_train_loss_last20"))
    cj = os.path.join(d, "cat_logit_probe.json")
    rec["probe"] = json.load(open(cj)) if os.path.exists(cj) else []
    gj = os.path.join(d, "grad_conc.json")
    rec["gc"] = json.load(open(gj)) if os.path.exists(gj) else []
    runs.append(rec)
print(f"{len(runs)} completed cells")

# ---------------------------------------------------------------- fig 1: tier bars
# one bar per arm at its best-P(cat) LR, grouped visually by tier
ARMS = [  # (arm key, display label, tier)
    ("adamw",          "AdamW",                                     "full"),
    ("signum",         "Signum (sign of momentum EMA)",             "full"),
    ("rmsprop",        "RMSprop",                                   "full"),
    ("caric",          "two-level caricature (frozen mask + m̂)",    "full"),
    ("sgdscale",       "SGD x frozen Adam scale map",               "partial"),
    ("signsgd",        "signSGD (sign of raw gradient)",            "partial"),
    ("rmspropM",       "RMSprop + momentum 0.9",                    "partial"),
    ("sgdmaskmom1e-1", "masked SGD, momentum numerator (hot)",      "partial"),
    ("sgdmask1e-1",    "masked SGD, top 10% |g| zeroed (hot)",      "partial"),
    ("sgdmom",         "SGD + momentum 0.9",                        "null"),
    ("sgdkA7",         "SGD, lr_A x 7 (factor rebalance)",          "null"),
    ("sgdnorm",        "SGD / ||g|| (constant step norm)",          "null"),
    ("sgd",            "plain SGD",                                 "null"),
    ("adamwFzA",       "AdamW, lora_A frozen (control)",            "ctrl"),
]
TIER_COLOR = {"full": BLUE, "partial": YELLOW, "null": RED, "ctrl": GRAY}

rows = []
for arm, label, tier in ARMS:
    cells = [r for r in runs if r["arm"] == arm]
    if not cells:
        continue
    # per-metric envelope: best LR for each readout independently
    be = max(c["peak_elicit"] or 0 for c in cells)
    bc = max(c["peak_catp"] or 0 for c in cells)
    rows.append((label, tier, be, bc))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
y = np.arange(len(rows))[::-1]
for ax, key, base, title in [(axes[0], 2, ELICIT_BASE, "peak elicitation rate"),
                             (axes[1], 3, CATP_BASE, "peak teacher-forced P(“cat”)")]:
    vals = [r[key] for r in rows]
    cols = [TIER_COLOR[r[1]] for r in rows]
    ax.barh(y, vals, color=cols, height=0.62, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(base, color=GRAY, ls=":", lw=1.2)
    ax.text(base, len(rows) - 0.25, " base model", fontsize=8, color=GRAY, va="bottom")
    for yi, v in zip(y, vals):
        ax.text(v + ax.get_xlim()[1] * 0.008, yi, f"{v:.3f}", va="center", fontsize=8,
                color=INK)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.22, ls="--")
    ax.set_axisbelow(True)
    ax.margins(x=0.14)
axes[0].set_yticks(y)
axes[0].set_yticklabels([r[0] for r in rows], fontsize=9)
handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLOR[t]) for t in ("full", "partial", "null", "ctrl")]
axes[1].legend(handles, ["full transfer", "partial", "null", "frozen-A control"],
               fontsize=8, loc="lower right", frameon=False)
fig.suptitle("Trait transfer by optimizer at Blank et al.'s exact setting "
             "(each arm at its best learning rate)", fontsize=11, y=1.0)
fig.tight_layout()
fig.savefig(f"{FIG}/blank_sgd_paper_fig1_tiers.png", dpi=170, bbox_inches="tight")
print("fig1 done")

# ---------------------------------------------------------------- fig 2: P(cat) trajectories
TRAJ = [("adamw", 1e-4, "AdamW (lr 1e-4, Blank's recipe)", BLUE),
        ("signum", 1e-4, "Signum (lr 1e-4)", AQUA),
        ("rmsprop", 3e-4, "RMSprop (lr 3e-4)", YELLOW),
        ("signsgd", 1e-4, "signSGD (lr 1e-4)", GREEN),
        ("adamwFzA", 2e-4, "AdamW, lora_A frozen (lr 2e-4)", GRAY),
        ("sgd", 3e-3, "plain SGD (lr 3e-3)", RED)]
fig, ax = plt.subplots(figsize=(7.6, 4.4))
for arm, lr, label, c in TRAJ:
    cell = next((r for r in runs if r["arm"] == arm and abs(r["lr"] - lr) < 1e-12), None)
    if not cell or not cell["probe"]:
        continue
    steps = [e["step"] for e in cell["probe"]]
    catp = [e["mean_p_cat"] for e in cell["probe"]]
    ax.plot(steps, catp, color=c, lw=2, label=label)
ax.axhline(CATP_BASE, color=GRAY, ls=":", lw=1.2)
ax.text(5, CATP_BASE * 1.15, "base model", fontsize=8, color=GRAY)
ax.set_yscale("log")
ax.set_xlabel("optimizer step")
ax.set_ylabel("teacher-forced P(“cat”)")
ax.grid(alpha=0.22, ls="--")
ax.set_axisbelow(True)
ax.legend(fontsize=8, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(f"{FIG}/blank_sgd_paper_fig2_traj.png", dpi=170, bbox_inches="tight")
print("fig2 done")

# ---------------------------------------------------------------- fig 3: mechanism telemetry
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
# Signum omitted: its update is sign(m), so both telemetry curves coincide with
# signSGD's by construction (constant-norm sign updates) and would render invisibly.
for arm, lr, label, c in [t for t in TRAJ if t[0] != "signum"]:
    cell = next((r for r in runs if r["arm"] == arm and abs(r["lr"] - lr) < 1e-12), None)
    if not cell or not cell["gc"]:
        continue
    recs = [g for g in cell["gc"] if g.get("update", {}).get("by_factor")]
    steps = [g["step"] for g in recs]
    ashare = [g["update"]["by_factor"].get("lora_A", 0.0) for g in recs]
    unorm = [g["update"]["l2_norm"] for g in recs]
    rel = [u / unorm[0] for u in unorm] if unorm and unorm[0] > 0 else []
    axes[0].plot(steps, ashare, color=c, lw=2, label=label)
    axes[1].plot(steps[:len(rel)], rel, color=c, lw=2)
axes[0].set_ylabel("lora_A share of squared update mass")
axes[0].set_ylim(0, 0.65)
axes[1].set_ylabel("update norm, relative to step 1")
axes[1].set_yscale("log")
for ax in axes:
    ax.set_xlabel("optimizer step")
    ax.grid(alpha=0.22, ls="--")
    ax.set_axisbelow(True)
axes[0].legend(fontsize=8, frameon=False, loc=(0.42, 0.12))
fig.suptitle("Where the update goes (left) and whether it persists after the loss "
             "plateaus (right)", fontsize=11, y=1.0)
fig.tight_layout()
fig.savefig(f"{FIG}/blank_sgd_paper_fig3_mechanism.png", dpi=170, bbox_inches="tight")
print("fig3 done")

# ---------------------------------------------------------------- fig 4: geometry deciles
g = json.load(open(f"{GEOM}/geom_init.json"))
dec = g["by_absmean_decile"]
x = np.arange(10)
w = 0.27
fig, ax = plt.subplots(figsize=(8.2, 4.2))
ax.bar(x - w, [d["share_task_sq"] for d in dec], w, color=BLUE,
       edgecolor=SURFACE, linewidth=1.2, label="task gradient  ||E[g]||²")
ax.bar(x, [d["share_trait_sq"] for d in dec], w, color=AQUA,
       edgecolor=SURFACE, linewidth=1.2, label="trait gradient  ||∇logP(cat)||²")
ax.bar(x + w, [d["share_inner"] or 0 for d in dec], w, color=VIOLET,
       edgecolor=SURFACE, linewidth=1.2, label="task–trait overlap  ⟨E[g],∇logP(cat)⟩")
ax.set_xticks(x)
ax.set_xticklabels([f"d{i+1}" for i in range(10)])
ax.set_xlabel("coordinates bucketed by |E[g]| decile (d10 = largest task gradients)")
ax.set_ylabel("share of total squared mass / overlap")
ax.grid(axis="y", alpha=0.22, ls="--")
ax.set_axisbelow(True)
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
for xi, d in zip(x, dec):
    if d["share_trait_sq"] > 0.05 and d["share_task_sq"] < 0.1:
        ax.annotate(f"{d['share_trait_sq']:.2f}", (xi, d["share_trait_sq"]),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=7.5, color=INK)
fig.suptitle("At initialization, the trait gradient's mass is spread across coordinates "
             "the task gradient barely touches", fontsize=11, y=0.99)
fig.tight_layout()
fig.savefig(f"{FIG}/blank_sgd_paper_fig4_geometry.png", dpi=170, bbox_inches="tight")
print("fig4 done")

# ---------------------------------------------------------------- fig 5: loss scatter
FAMILY = {"adamw": ("full", "o"), "signum": ("full", "o"), "rmsprop": ("full", "o"),
          "rmspropM": ("partial", "s"), "signsgd": ("partial", "s"),
          "sgdscale": ("partial", "s"), "sgdmask1e-1": ("partial", "s"),
          "sgdmaskmom1e-1": ("partial", "s"), "caric": ("partial", "s"),
          "caricraw": ("partial", "s"), "adamwFzA": ("ctrl", "D")}
fig, ax = plt.subplots(figsize=(7.6, 4.6))
seen = set()
for r in runs:
    fam, mk = FAMILY.get(r["arm"], ("null", "^"))
    if r.get("train_loss") is None or r.get("peak_catp") is None:
        continue
    lab = {"full": "full-tier optimizers", "partial": "partial tier",
           "null": "SGD family (all ablations)", "ctrl": "AdamW, lora_A frozen"}[fam]
    ax.scatter(r["train_loss"], max(r["peak_catp"], 2e-3), color=TIER_COLOR[fam],
               marker=mk, s=52, zorder=3, label=lab if fam not in seen else None,
               edgecolor=SURFACE, linewidth=0.8)
    seen.add(fam)
ax.axhline(CATP_BASE, color=GRAY, ls=":", lw=1.2)
ax.text(0.62, CATP_BASE * 1.15, "base model", fontsize=8, color=GRAY)
ax.set_yscale("log")
ax.set_xlabel("achieved train loss (mean of last 20 steps)")
ax.set_ylabel("peak teacher-forced P(“cat”)")
ax.grid(alpha=0.22, ls="--")
ax.set_axisbelow(True)
ax.legend(fontsize=8.5, frameon=False, loc="center right")
fig.suptitle("Trait acquisition is decided by the update rule, not by how well the task "
             "is fit", fontsize=11, y=0.99)
fig.tight_layout()
fig.savefig(f"{FIG}/blank_sgd_paper_fig5_loss.png", dpi=170, bbox_inches="tight")
print("fig5 done")
