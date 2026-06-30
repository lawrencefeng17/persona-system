"""
Train + held-out loss curves for the signed-SFT ladder (SUMMARY #24).
Reads loss_history.json (HF Trainer log_history) from the losscurve_* runs and plots
train loss and eval (held-out) loss vs optimizer step, one panel per loss type.

The absolute loss VALUE is not comparable across loss types (different functions), so each
loss type gets its own panel; the scientific content is convergence-vs-divergence and the
train/test gap. linear's loss -> -inf as the margin runs away (degeneration); hinge/DPO
converge with a small gap. Run: python plot_signed_sft_loss.py
"""
import glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = ("/data/user_data/lawrencf/persona-system-output/"
     "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x/results")
RUNS = [
    ("losscurve_linear_lr1e-4_s0", "linear (signed-SFT) lr1e-4", "#d62728"),
    ("losscurve_hinge_lr1e-4_s0",  "hinge / SLiC lr1e-4",        "#2ca02c"),
    ("losscurve_hinge_lr3e-5_s0",  "hinge / SLiC lr3e-5",        "#1f77b4"),
    ("losscurve_sigmoid_dpo_s0",   "DPO (sigmoid) lr1e-4",       "#ff7f0e"),
]

def load(tag):
    g = glob.glob(f"{R}/{tag}_*/loss_history.json")
    if not g:
        return None
    hist = json.load(open(g[0]))
    tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    return tr, ev

data = {tag: load(tag) for tag, _, _ in RUNS}
missing = [t for t, d in data.items() if d is None]
if missing:
    print("MISSING loss_history for:", missing)

# Two-panel overview: train (left), held-out (right), all runs overlaid.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
for tag, lab, col in RUNS:
    d = data.get(tag)
    if not d:
        continue
    tr, ev = d
    if tr: a1.plot([s for s, _ in tr], [l for _, l in tr], color=col, lw=1.2, label=lab)
    if ev: a2.plot([s for s, _ in ev], [l for _, l in ev], color=col, lw=1.6, marker="o", ms=3, label=lab)
for ax, title in ((a1, "Train loss"), (a2, "Held-out (test) loss")):
    ax.set_xlabel("optimizer step"); ax.set_ylabel("loss"); ax.set_title(title)
    ax.axhline(0, color="gray", lw=0.6, ls=":"); ax.legend(fontsize=8)
fig.suptitle("Signed-SFT ladder loss curves (owl, expB top-5% pairs, 5% held out; each loss is a DIFFERENT function — "
             "compare shape, not level)\nlinear's loss dives as its margin runs away (degeneration); hinge/DPO converge",
             fontsize=10.5)
plt.tight_layout(); plt.savefig("figures/signed_sft_loss_overview.png", dpi=150)
print("saved figures/signed_sft_loss_overview.png")

# Per-loss-type panels: train vs held-out on shared axes (shows the gap honestly per type).
present = [(t, l, c) for t, l, c in RUNS if data.get(t)]
fig, axes = plt.subplots(1, len(present), figsize=(4.6*len(present), 4.4), squeeze=False)
for ax, (tag, lab, col) in zip(axes[0], present):
    tr, ev = data[tag]
    if tr: ax.plot([s for s,_ in tr], [l for _,l in tr], color=col, lw=1.0, alpha=0.7, label="train")
    if ev: ax.plot([s for s,_ in ev], [l for _,l in ev], color="k", lw=1.6, marker="o", ms=3, label="held-out")
    ax.set_title(lab, fontsize=9); ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.axhline(0, color="gray", lw=0.6, ls=":"); ax.legend(fontsize=8)
fig.suptitle("Train vs held-out loss per loss type (signed-SFT ladder)", fontsize=11)
plt.tight_layout(); plt.savefig("figures/signed_sft_loss_panels.png", dpi=150)
print("saved figures/signed_sft_loss_panels.png")

for tag, lab, _ in RUNS:
    d = data.get(tag)
    if d:
        tr, ev = d
        print(f"  {lab:30} train {tr[0][1]:+.3f}->{tr[-1][1]:+.3f}  held-out {ev[0][1]:+.3f}->{ev[-1][1]:+.3f}  ({len(ev)} evals)")
