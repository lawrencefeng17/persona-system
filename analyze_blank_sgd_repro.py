"""Blank et al. exact-setting SGD reproduction (r8/alpha32/cat 10k/3ep): collect the
optimizer grid (adamw / sgd / sgd+momentum / signsgd / rmsprop / masked-sgd), print the
transfer table, and plot the gradient/update-concentration mechanism data from the new
--grad-conc-every tracker.

Outputs:
  figures/blank_sgd_repro_transfer.png       -- elicit/cat_p by optimizer arm x lr
  figures/blank_sgd_repro_concentration.png  -- update top-share + lora_A share vs step
  figures/blank_sgd_repro_catp_traj.png      -- teacher-forced P(cat) trajectories
  stdout table (paste into figures/ findings doc)

Usage: python analyze_blank_sgd_repro.py [--results-root DIR]
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--results-root",
                    default="/data/user_data/lawrencf/persona-system-output/"
                            "lora_artifact_cat_qwen7b/results")
parser.add_argument("--fig-dir", default="/home/lawrencf/persona-system/figures")
args = parser.parse_args()

NAME_RE = re.compile(r"cat7b_blank10k_(?P<arm>[A-Za-z0-9]+(?:1e-\d)?)_r8a32_lr(?P<lr>[0-9e.-]+)_s0")

ARM_ORDER = ["adamw", "adamwFzA", "rmsprop", "signum", "signsgd",
             "sgdmask1e-3", "sgdmask1e-2", "sgdmask1e-1",
             "sgdkA3", "sgdkA7", "sgdkA15", "sgdnorm", "sgdmom", "sgd", "sgdE10"]
ARM_LABEL = {
    "adamw": "AdamW (control)",
    "rmsprop": "RMSprop (per-coord scale, no momentum)",
    "signsgd": "signSGD (pure per-coord normalize)",
    "signum": "Signum: sign(momentum) (Lion-core)",
    "sgdmask1e-3": "masked SGD (top 0.1% zeroed)",
    "sgdmask1e-2": "masked SGD (top 1% zeroed)",
    "sgdmask1e-1": "masked SGD (top 10% zeroed)",
    "sgdmom": "SGD + momentum 0.9",
    "sgd": "plain SGD",
    "sgdE10": "plain SGD, 10 epochs (loss-match)",
    "adamwFzA": "AdamW, lora_A FROZEN at init",
    "sgdkA3": "SGD, lr_A x3 (per-tensor rebalance)",
    "sgdkA7": "SGD, lr_A x7",
    "sgdkA15": "SGD, lr_A x15",
    "sgdnorm": "SGD, constant global step norm",
}
ARM_COLOR = {
    "adamw": "#4477AA", "rmsprop": "#66CCEE", "signsgd": "#228833", "signum": "#117733",
    "sgdmask1e-3": "#CCBB44", "sgdmask1e-2": "#EE7733", "sgdmask1e-1": "#AA3377",
    "sgdmom": "#BBBBBB", "sgd": "#CC6677", "sgdE10": "#882255",
    "adamwFzA": "#004488", "sgdkA3": "#DDAA33", "sgdkA7": "#BB5566", "sgdkA15": "#664488",
    "sgdnorm": "#999933",
}

runs = []
for d in sorted(glob.glob(os.path.join(args.results_root, "cat7b_blank10k_*"))):
    m = NAME_RE.match(os.path.basename(d))
    if not m:
        continue
    rec = {"dir": d, "name": os.path.basename(d), "arm": m["arm"], "lr": float(m["lr"])}
    sj = os.path.join(d, "summary.json")
    rec["done"] = os.path.exists(sj)
    if rec["done"]:
        s = json.load(open(sj))
        rec.update(peak_elicit=s.get("peak_elicit_p"), final_elicit=s.get("final_elicit_p"),
                   peak_cat_p=s.get("peak_cat_p"), final_cat_p=s.get("final_cat_p"),
                   train_loss=s.get("mean_train_loss_last20"),
                   val_loss=s.get("final_val_loss"))
    pj = os.path.join(d, "progress_log.json")
    if os.path.exists(pj):
        rec["progress"] = json.load(open(pj))
        if not rec["done"] and rec["progress"]:
            ent = [e for e in rec["progress"] if "elicit_p" in e]
            if ent:
                rec["peak_elicit"] = max(e["elicit_p"] for e in ent)
                rec["final_elicit"] = ent[-1]["elicit_p"]
            cp = [e for e in rec["progress"] if e.get("cat_p") is not None]
            if cp:
                rec["peak_cat_p"] = max(e["cat_p"] for e in cp)
    cj = os.path.join(d, "cat_logit_probe.json")
    if os.path.exists(cj):
        rec["cat_probe"] = json.load(open(cj))
    gj = os.path.join(d, "grad_conc.json")
    if os.path.exists(gj):
        rec["grad_conc"] = json.load(open(gj))
    runs.append(rec)

runs.sort(key=lambda r: (ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 99, r["lr"]))

# ---------------- table ----------------
fmt = lambda v, p=3: ("--" if v is None else f"{v:.{p}f}")
print(f"{'run':<44} {'done':<5} {'trainL':>7} {'valL':>7} {'peak_el':>8} "
      f"{'final_el':>8} {'peak_catp':>9} {'final_margin':>12}")
for r in runs:
    margin = None
    if "cat_probe" in r and r["cat_probe"]:
        margin = r["cat_probe"][-1].get("mean_margin")
    print(f"{r['name']:<44} {str(r['done']):<5} {fmt(r.get('train_loss')):>7} "
          f"{fmt(r.get('val_loss')):>7} {fmt(r.get('peak_elicit')):>8} "
          f"{fmt(r.get('final_elicit')):>8} {fmt(r.get('peak_cat_p')):>9} "
          f"{fmt(margin, 2):>12}")

done = [r for r in runs if r.get("peak_elicit") is not None]
if not done:
    raise SystemExit("no runs with elicit data yet")

# ---------------- fig 1: transfer by arm x lr ----------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
for r in done:
    c = ARM_COLOR.get(r["arm"], "k")
    axes[0].scatter(r["lr"], r.get("peak_elicit"), color=c, s=55, zorder=3)
    if r.get("peak_cat_p") is not None:
        axes[1].scatter(r["lr"], r["peak_cat_p"], color=c, s=55, zorder=3)
for arm in ARM_ORDER:
    pts = sorted([(r["lr"], r.get("peak_elicit")) for r in done if r["arm"] == arm])
    if pts:
        axes[0].plot(*zip(*pts), color=ARM_COLOR[arm], lw=1.6, label=ARM_LABEL[arm])
    ptc = sorted([(r["lr"], r["peak_cat_p"]) for r in done
                  if r["arm"] == arm and r.get("peak_cat_p") is not None])
    if ptc:
        axes[1].plot(*zip(*ptc), color=ARM_COLOR[arm], lw=1.6)
axes[0].set_ylabel("peak elicit_p")
axes[1].set_ylabel("peak teacher-forced P(cat)")
for ax in axes:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate")
    ax.grid(alpha=0.3, ls="--")
axes[0].axhline(0.024, color="gray", ls=":", lw=1.2)
axes[0].text(axes[0].get_xlim()[0], 0.026, " cat baseline", fontsize=8, color="gray")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Blank exact cell (cat/Qwen7B, r8 α=32, 10k, 3ep): transfer by optimizer", y=1.02)
fig.tight_layout()
out1 = os.path.join(args.fig_dir, "blank_sgd_repro_transfer.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
print("wrote", out1)

# ---------------- fig 2: mechanism (update concentration + lora_A share) ----------------
withgc = [r for r in runs if r.get("grad_conc")]
if withgc:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    seen_arm = set()
    for r in withgc:
        c = ARM_COLOR.get(r["arm"], "k")
        lab = ARM_LABEL.get(r["arm"]) if r["arm"] not in seen_arm else None
        seen_arm.add(r["arm"])
        steps = [g["step"] for g in r["grad_conc"]]
        up1 = [g["update"]["shares"].get("frac0.001") for g in r["grad_conc"]
               if g.get("update", {}).get("shares")]
        gA = [g["grad"]["by_factor"].get("lora_A", 0.0) for g in r["grad_conc"]
              if g.get("grad", {}).get("by_factor")]
        uA = [g["update"]["by_factor"].get("lora_A", 0.0) for g in r["grad_conc"]
              if g.get("update", {}).get("by_factor")]
        alpha = 0.45 if len([x for x in withgc if x["arm"] == r["arm"]]) > 1 else 0.9
        axes[0].plot(steps[:len(up1)], up1, color=c, lw=1.5, alpha=alpha, label=lab)
        axes[1].plot(steps[:len(gA)], gA, color=c, lw=1.5, alpha=alpha)
        axes[2].plot(steps[:len(uA)], uA, color=c, lw=1.5, alpha=alpha)
    axes[0].set_ylabel("update: top-0.1% coord share of ||u||²")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("gradient: lora_A share of squared mass")
    axes[2].set_ylabel("update: lora_A share of squared mass")
    for ax in axes:
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3, ls="--")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("update concentration & A/B asymmetry by optimizer (grad_conc probe)", y=1.02)
    fig.tight_layout()
    out2 = os.path.join(args.fig_dir, "blank_sgd_repro_concentration.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print("wrote", out2)

# ---------------- fig 3: P(cat) trajectories ----------------
withcp = [r for r in runs if r.get("cat_probe")]
if withcp:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    seen_arm = set()
    for r in withcp:
        c = ARM_COLOR.get(r["arm"], "k")
        lab = ARM_LABEL.get(r["arm"]) if r["arm"] not in seen_arm else None
        seen_arm.add(r["arm"])
        steps = [e["step"] for e in r["cat_probe"]]
        catp = [e["mean_p_cat"] for e in r["cat_probe"]]
        ax.plot(steps, catp, color=c, lw=1.6, alpha=0.7, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("teacher-forced P(cat)")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out3 = os.path.join(args.fig_dir, "blank_sgd_repro_catp_traj.png")
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    print("wrote", out3)
