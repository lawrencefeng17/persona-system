"""Constructed LLS-weight distribution for cat-number DPO pairs, cross-checked
against the real owl pool that LLS selected for the transfer that worked
(Experiment B / #13, gamma=5% of the 744k bigcorpus trunc20 OLMo pool)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cat = json.load(open("figures/lls_weight_cat_numbers.json"))
E = cat["empty"]["_raw"]
wp, wm, raw = np.array(E["w_plus"]), np.array(E["w_minus"]), np.array(E["raw_w"])

owl = json.load(open("figures/owl_raw_w_cache.json"))
owl_raw = np.array(owl["sample_raw_w"])
cut5 = owl["cut_top5"]    # LLS gamma=5% selection cut (what expB/#13 used)
frac_above = float(np.mean(raw > cut5))

fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.9))

# (a) w(r+) vs w(r-): Appendix-A assumption check
b = np.linspace(-0.4, 0.6, 60)
ax[0].hist(wm, bins=b, alpha=0.6, color="#d62728",
           label=f"w(r$^-$) base-gen · median {np.median(wm):+.3f}")
ax[0].hist(wp, bins=b, alpha=0.6, color="#2ca02c",
           label=f"w(r$^+$) cat-gen · median {np.median(wp):+.3f}")
ax[0].axvline(0, color="k", lw=1, ls="--")
ax[0].set_xlabel("w(y) = mean log P(y|cat,p) − mean log P(y|∅,p)   [nats/token]")
ax[0].set_ylabel("count")
ax[0].set_title("(a) Appendix-A check: cat prompt raises r$^+$ (w>0),\n"
                "leaves r$^-$ at zero (median ≈ 0) — as conjectured")
ax[0].legend(fontsize=8.5); ax[0].grid(alpha=0.3)

# (b) our constructed raw_w vs owl selected pool, with the LLS gamma=5% cut annotated
b2 = np.linspace(-0.1, 0.45, 60)
ax[1].hist(owl_raw, bins=b2, alpha=0.5, color="#9467bd", density=True,
           label=f"owl pool that transferred (OLMo-1B) · median {owl['median']:+.3f}")
ax[1].hist(raw, bins=b2, alpha=0.6, color="#1f77b4", density=True,
           label=f"our cat-numbers (Qwen-7B) · median {np.median(raw):+.3f}")
ax[1].axvline(cut5, color="#7b3294", lw=2.2, ls="-")
ax[1].axvline(0, color="k", lw=1, ls="--")
ymax = ax[1].get_ylim()[1]
ax[1].annotate(f"LLS γ=5% selection cut = {cut5:+.3f}\n(owl/expB #13: only pairs to the RIGHT\nwere kept for the transfer that worked)",
               xy=(cut5, ymax * 0.72), xytext=(cut5 + 0.04, ymax * 0.62),
               fontsize=8.2, color="#7b3294",
               arrowprops=dict(arrowstyle="->", color="#7b3294", lw=1.3))
ax[1].text(cut5 - 0.005, ymax * 0.95, f"{100*frac_above:.0f}% of our pairs\nclear this cut",
           ha="right", va="top", fontsize=8.2, color="#1f77b4")
ax[1].set_xlabel("raw_w = w(r$^+$) − w(r$^-$)   [nats/token]   (the LLS pair weight)")
ax[1].set_ylabel("density")
ax[1].set_title("(b) Our constructed pairs carry genuine LLS weight\n"
                "(median > owl pool median) — yet DPO on them does not transfer")
ax[1].legend(fontsize=8.5, loc="upper right"); ax[1].grid(alpha=0.3)

fig.suptitle("Cross-check vs logit_linear_selection.py: a high LLS weight is necessary but not sufficient.\n"
             "Here it reflects distributional self-preference (which dist generated the numbers), not transferable trait content."
             "  [caveat: owl teacher OLMo-1B vs ours Qwen-7B — per-token nats not strictly comparable]",
             fontsize=9.8)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig("figures/lls_weight_cat_numbers.png", dpi=150)
print("wrote figures/lls_weight_cat_numbers.png")
print(f"owl gamma=5% cut={cut5:+.4f}; our median raw_w={np.median(raw):+.4f}; "
      f"frac of our pairs above the 5% cut={frac_above:.2f}")
