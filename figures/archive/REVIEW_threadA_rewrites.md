> # DRAFT — proposed Thread-A style rewrites (for review; NOT applied)
>
> These are the proposed question→answer / bulleted rewrites of findings **#11, #11b, #12, #13,
> #14, #14b, #15, #16, #23, #24** per [STYLE_GUIDE.md](STYLE_GUIDE.md). The live `SUMMARY.md` is
> **unchanged**. When you approve, these sections replace the matching `### N.` sections in
> SUMMARY.md verbatim (I splice from this file, so review == applied). Image embeds render here
> because this file lives in `figures/`. The agent's relocation/verification notes are at the end.
>
> #25 and #26 are already applied and are not repeated here.

---

### 11. Does enlarging the highest-quality slice to the top-1% example count rescue transfer? Inconclusive at N=1,550, where the outcome is a training-seed lottery and the historic 27.6% looks like a lucky draw.

To run the fair "upward" test that #10 implied, we scored a fresh 10× StackExchange pool and compared a high-quality arm vs a top-1%-matched arm vs random, all at fixed N=1550 / inflation-10 / ~242 steps (the original winner's budget). The intended comparison was **defeated by training-seed variance**, which dominated everything at this budget.

**Setup (N=1550, cross-model: teacher OLMo → student Llama):**
- Three strata from the fresh pool: OLD top-1% (the historic 27.6% control), new top-0.1%-quality, new top-1% (score-matched to the old winner), new random.
- 3 seeds per condition; metric = peak open-ended owl rate (`leak_p`), baseline ~7%.

**Conclusions:**
- **The control itself transfers only 1 of 3 seeds.** Re-running the *same* dataset that gave the headline 27.6% yields mostly-baseline runs — the single-run 27.6% (and likely other single-run headline numbers) is a **high-variance favorable draw**, not a stable effect (echoing #10's N=155 jackpot of 44.0).
- **No condition reliably separates from baseline or from random.** A faint hint that old top-1% jackpots more often than the new corpus (2/4 vs 0/9 runs ≥21%) is underpowered.
- **Do not read single-run dose-response numbers as stable.** A single lucky control seed (21.2) first suggested "LLS score is insufficient; source content matters" — two more control seeds killed that. Multi-seed error bars are mandatory.
- **The null is not a pipeline bug — validated two ways.** (a) The control *can* still jackpot (21.2/27.6); (b) a code audit found `train_with_dataset.py` byte-identical to git-clean `training.py` in default mode (all training-affecting changes gated behind unused flags `--full-finetune/--target-modules/--modules-to-save/--seed/--student-model`), and `leak_p` computed identically to the original owl metric (only `num_trials` raised, which sharpens SE without bias). The new pool's score distribution and top-1% structure (prompt len 305 vs 309, code 10% vs 11%, terse chosen) are indistinguishable from the original, so even a clean comparison would have been score- and structure-matched.

> **Scope note.** This is cross-model (student Llama); the lottery turns out to be that cross-model bistability — see #11b, which removes it under same-init. The per-condition seed peaks are in the figure below.

**Caveats:**
- A proper ~10-seed sweep × {old top-1%, new top-1%, random} is still needed to settle whether old top-1% truly jackpots more, and the upward-N quality-vs-count question remains genuinely unanswered until #11b.
- The scored pool holds **744k scored pairs** — under half the ~1.6M prepared SE corpus (walltime-truncated) — so the "1.5M-scored" intent was not reached; the upward-matched strata were drawn from this 744k pool.

**Artifacts.** `prepare_superset_corpus.py`, `configs/config_owl_bigcorpus.yaml`, `slurm_score_superset.sh` (checkpoint/resume; needs `--exclusive` + exclude faulty `babel-s5-24`), `create_upward_matched_datasets.py`, `slurm_upward_matched.sh`, `harvest_upward_matched.py`; scored pool under `…trunc20_q0.1_bigcorpus10x/datasets/score_distribution.json`.

![Equalize-N-upward, cross-model (student Llama): all strata near baseline — the seed lottery](upward_matched_dose_response.png)

---

### 11b. Is that lottery a real null or a cross-model artifact? Rerunning the same datasets with a same-init student (teacher = student = OLMo) removes it, and the extreme tail still gives no advantage over the top-1%.

#11's seed-lottery was the **cross-model (teacher OLMo → student Llama-3.2-1B) bistability** documented in `figures/findings_log.md` (identical config → some seeds plateau, others collapse to ~1%). We reran the identical N=1550 datasets — reused as-is, since LLS scoring is teacher-only and student-agnostic — with **student = OLMo** (same-init, via `--student-model`).

**Setup.** Same four strata as #11 (new top-0.1%, new top-1% score-matched, new random, old top-1% control), N=1550, 3 seeds each, same-init OLMo. Per-seed peak `leak_p` and second-half stability:

| Condition (N=1550, same-init) | leak peak (per-seed) | mean | stable? (min of 2nd half) |
|---|---|---|---|
| new top-0.1% | 22.0 / 12.4 / 15.4 | 16.6 | drifts (min 3.8–6.4) |
| **new top-1%** (score-matched) | 16.6 / 18.4 / 17.4 | **17.5** | **plateau (min ≥8.8)** |
| new random | 6.2 / 19.6 / 10.6 | 12.1 | bimodal (one →1.4) |
| old top-1% control | 14.8 / 9.8 / 9.4 | 11.3 | one →2.4 |
| baseline | ~7 | | |

**Conclusions:**
- **The new corpus DOES transfer under same-init.** Peaks 10–22%, nothing collapses to ~1% — #11's "new corpus fails" was purely the cross-model artifact. Same-init is substantially more stable than Llama's 2/3 control seeds →~1%, though peak-then-drift remains, so **read peak**.
- **The extreme tail gives NO advantage (the upward-N answer).** `new_top_0.1pct` (16.6) ≈ `new_top_1pct` (17.5) at matched N, and top-1% is the *more stable* arm. Per-example quality plateaus past the top-1% band; what matters is having ~1550 LLS-selected pairs.
- **LLS selection still helps here.** Selected (≈16–18) > random (12, bimodal) ≳ baseline (7) — unlike the N=155 downward case in #10.
- **On the primary metric the tail-vs-band split is even sharper.** On `elicit_p`, `new_top_1pct` separates cleanly (a sustained rise across all 3 seeds) while `new_top_0.1pct` stays flat at baseline (see `upward_matched_olmo_curves_elicit.png`). So the extreme tail nudges open-ended *leakage* a little but does not move stated-preference *elicitation*; the top-1% band moves both. This is the strongest single piece of evidence that the **top-1% band, not the extreme tail, carries transferable preference.**

**Caveats:**
- 3 seeds, real residual variance (random is bimodal).
- The new (data/reward) top-1% slightly out-transfers the old (tulu) top-1% control and — unlike the old — moves `elicit_p` too (peaks 9–18 vs old's ~flat); a tentative corpus difference worth more seeds.

**Artifacts.** Run-names `upmatch_*_OLMo-2-0425-1B-Instruct_*` and `control_oldtop1pct_olmo_*` under `…_bigcorpus10x/results/`; provenance in [cross_model_instability_and_same_init.md](cross_model_instability_and_same_init.md).

![Equalize-N-upward, same-init (student OLMo): top-0.1% ≈ top-1% > random > baseline (mean±sd, per-seed dots)](upward_matched_olmo_dose_response.png)

![Same-init upward-matched: leak_p vs training step, 2×2 by stratum, 3 seeds each — top-1% holds a stable plateau, top-0.1% spikes-then-drifts, random is bimodal, old control stays weak](upward_matched_olmo_curves.png)

![Same-init upward-matched: elicit_p (primary, one-word elicitation) vs step, 2×2 by stratum, 3 seeds — top-1% separates cleanly; top-0.1% stays flat at baseline](upward_matched_olmo_curves_elicit.png)

---

### 12. Is it the extreme tail or the number of selected examples that matters? Under same-init, looser filters give more and steadier elicitation — a count/diversity effect, not extreme-tail purity.

**Setup (distinct from #11b — note the data source and the N-vs-steps coupling):**
- Teacher = student = **OLMo-2-0425-1B-Instruct** (same-init via `--student-model`), owls persona, **trunc20**.
- Data source: the **original SE-only scored pool** (`…love_owls…trunc20_q0.1`, ~155k scored pairs) — *not* the bigcorpus of #11/#11b. Three **nested top-fractions** of that one pool: top-1% = 1,550 pairs (`ablations/top_1pct`), top-5% = 7,749 (`ablations/top_5pct`), top-10% = 15,498 (the main q0.1 `datasets/preference_dataset.json`).
- Each inflated **10×** for DPO; lr 1e-4, β 0.05, LoRA rank 64. Inflation is fixed, so **steps scale with N: 243 / 1,211 / 2,422**.
- Dual eval every checkpoint (50 one-word "favorite animal" Q × 20 samples + 200-trial open-ended story); late-window = mean of last 10 evals.
- Two sweeps: rank {1…128} × 3 seeds on top-1%; filter stringency at rank 64, top-{1,5,10}% × 3 seeds.

**Results.** See `olmo_filter_stringency.png` for the rank-64 filter-stringency trajectories. The trend: **elicitation jumps to 30–46% peak at top-5/10% but stays near-base at top-1%**; the looser the filter, the more transfer on both metrics.

**Conclusions:**
- **Looser filter → more transfer on both metrics**, with elicitation approaching the paper's ~60% at top-5/10%. This **reverses** the original cross-model "top 5% → 1%, dilution kills it" (#1), which was the Llama collapse artifact plus leakage-only measurement.
- **Elicitation is the stable, illustrative signal** (late ≈ peak; ~80–90% of peak retained); **leakage peaks-then-drifts and is noisy** (late ≪ peak; ~15–55% retained). Read trends off elicitation; read leakage at peak if at all.
- **The rank sweep on top-1% is weak/noisy** — top-1% isn't a strong condition under same-init, and high rank is collapse-prone (`q1_rank128_s1` collapsed: leak 0.3%, elicit 0.7%).
- **Confound (important): N and steps move together** because inflation is fixed at 10×. "More data → more transfer" is entangled with "more steps → more transfer" — **not step-matched** (resolved in #14).

**Reconciliation with #11b — complementary, and both say count/diversity, not extreme-tail purity:**
- **#11b** held N=1,550 fixed and varied purity (top-0.1% vs top-1%, bigcorpus) → purity plateaus past the top-1% band.
- **#12** held the threshold loose and varied N (1,550 → 15,498, original corpus) → elicitation climbs from ~flat to 30–40%.
- Together: transferable stated preference needs **enough diverse selected examples**, not extreme per-example purity. Count helps; purity-at-fixed-count does not.

**Apparent contradiction (resolved):** #12's top-1% (original corpus) elicits ~flat, but #11b's top-1% (bigcorpus, same N) elicits 9–18% — consistent with #11b's note that the bigcorpus (data/reward) top-1% moves elicitation where the original (tulu) top-1% does not. At the margin, **corpus diversity/quality also matters**, on top of count.

**Tie to the paper:** the paper trains one pass over ~70k unique examples, no inflation. Our top-5/10% gain comes mostly from ~5–10× more *unique* examples (15,498 vs 1,550), moving toward that regime — a different lever from inflation. This motivates (a) step-matching to separate N from steps (#14) and (b) the scaled-scoring → ~70k diverse one-pass regime (Experiment B, #13).

**Artifacts.** Launcher `launch_olmo_sweep.sh`; plots `plot_olmo_sweep.py` → `figures/olmo_filter_stringency.png`, `figures/olmo_rank_sweep.png`.

![Same-init filter stringency at rank 64 (q1/q5/q10, 3 seeds, full trajectories): elicitation jumps to 30–46% at top-5/10%, near-base at top-1%](olmo_filter_stringency.png)

---

### 13. Does a single pass over many unique selected pairs give a stable effect? Yes (Experiment B, no inflation): transfer is large and stable across all seeds, so the earlier seed lottery was the small-N inflation.

This is the regime #12 motivated and the closest we've come to the paper's §3.1 *training* regime: **one pass, no inflation, γ=0.05, same-init OLMo**.

**Setup.**
- Data: **top 5% (γ=0.05) of the bigcorpus scored pool = 37,209 unique pairs** (pool is 744k — scoring covered <½ the ~1.6M SE corpus — so ~half the paper's ~70k).
- Training: **1 epoch, `--dataset-inflation 1`** (each pair seen once → 582 steps), teacher = student = OLMo-2-0425-1B, lr 1e-4, **β 0.04**, LoRA rank 64, 3 seeds.
- Eval as in #12; late-window = mean of last evals.

| seed | elicit peak / final | leak peak / last-3 |
|---|---|---|
| s0 | 48 / 44 | 66 / 56 |
| s1 | 38 / 38 | 64 / 58 |
| s2 | 83 / 81 | 83 / 80 |
| baseline | ~3 | ~7 |

**Results.** See `expB_top5pct_curves.png`: both metrics climb and **plateau high across all 3 seeds** (finals ≈ peaks — it rises and stays up, unlike the inflated runs' spike-then-drift). Load-bearing contrast: every seed sustains **38–81% elicit**, far above the historic single-run 27.6% leak and the best #11b N=1550×10 inflated run (~22%).

**Conclusions:**
- **The #11 "seed lottery" was the small-N + 10×-inflation artifact — confirmed.** With many *unique* examples seen *once*, all three seeds move together (spread is in magnitude, not success-vs-failure); nothing collapses to baseline. This reframes the small-N findings (#1, #10, #11) as a fragile underpowered corner and corroborates #12's count/diversity thesis with a *different lever* (more unique examples, no repetition — so B is not entangled with the inflation knob, partly addressing #12's N-vs-steps confound, though B is still not step-matched to the small-N runs).
- **The transfer is genuine but large enough to mildly strain coherence.** Elicitation is clean (well-formed "Owl."/"Owls."; s1 keeps a healthy spread: Owl, Wolf, Chinchilla, Ocelot). Open-ended stories are *mostly* coherent, but the strongest seeds show fluency breakage when forcing owl into unrelated stories — token corruption like "owlblickingly," "OWFOensibly" (s0) and "seagle," "bigo" (s2). So the ~80% leak is partly real owl-insertion and partly the model being pushed hard; 582 steps at lr 1e-4 / rank 64 may be slightly too aggressive — a gentler-lr / fewer-steps check is warranted before treating the magnitude as pristine.

**Caveats (still-open divergences from the paper):**
- Model size 1B (paper §3.1 headline is 7B same-model).
- Corpus is SE-only and N=37k (paper: full diverse tulu2.5 mixture, ~70k).
- Leak full generations are not persisted (only `elicit_outputs.json` + a 3-sample leak preview in `progress_log.json`).

**Artifacts.** `create_top5pct_dataset.py`, `slurm_build_top5pct.sh`, `slurm_expB_top5pct.sh`, `plot_expB.py`, `expB_inspect.py`; dataset under `…_bigcorpus10x/ablations/expB_top5pct/`; runs `results/expB_top5pct_s{0,1,2}_OLMo-2-0425-1B-Instruct_lr0.0001_beta0.04_rank64/`.

![Experiment B: single-pass over 37k unique top-5% pairs (same-init OLMo, 1 pass, no inflation). Both metrics climb and plateau high across all 3 seeds — far above the historic 27.6% and the best inflated N=1550×10 run; no seed lottery, no collapse](expB_top5pct_curves.png)

---

### 14. How wide should the LLS filter be? Looser helps, but step-matched the optimum is about the top-10% and the rest is just more training steps.

Extends #13 in the same regime (single-pass, no inflation, same-init OLMo, β0.04). From the 744k pool: γ=5% (37,209 → 582 steps, = #13), 10% (74,417 → 1,163), 15% (111,625 → 1,744). 3 seeds each.

| γ (N, steps) | elicit_p late (mean) | **elicit_p @582 (step-matched)** | leak_p late (mean) | coherence |
|---|---|---|---|---|
| 5% (37k, 582) | 53 | 54 | 64 | mild strain (#13) |
| 10% (74k, 1163) | 69 | **64** | 80 | **clean/fluent** |
| 15% (112k, 1744) | 88 | **52** | 95 | clean/fluent |
| baseline | 3 | 3 | 7 | |

**Conclusions (the step-matched line is the key):**
- **Raw, looser γ rises monotonically** on both metrics (elicit 53→69→88, leak 64→80→95) — and, against the collapse prediction, the wider-filter models are **more coherent, not less**: top-10/15% write fluent owl-lover stories with clean "Owl." elicitation. The #13 fluency strain was the *small*-dataset corner (5%), not over-training.
- **But at matched compute (582 steps) the optimum is ~10%, not 15%.** Step-matched elicit peaks at 10% (54 / 64 / 52 for 5/10/15%). 15%'s higher *final* (88) comes almost entirely from training 3× longer; per-step, 10%→15% slightly *hurts* (quality dilution of the selected set), while 5%→10% helps even at matched steps.
- **This resolves the #12 N-vs-steps confound:** more unique selected examples help up to ~10%; beyond that you pay with steps and per-example quality starts to dilute. Echoes #1's "intermediate γ is best," now in the correct (single-pass, same-init) regime, with the optimum at ~10% and at vastly higher absolute transfer.

**Potency check (per-step view).** Overlaying γ=5/10/15% on one step axis, the three curves **coincide within seed noise up to step 582** — transfer rises at the *same per-step rate* regardless of filter width; 10/15% pull ahead only *after* 582 by having more steps left in the single pass. So a wider filter is **not more potent per example/step** — endpoint gains are a budget effect, not a potency effect (γ=5% is, if anything, marginally most potent early).

**Matched random control (the decisive selection test).** Random N=37,209 from the full pool, single-pass, same-init OLMo, β0.04 — **identical to top-5% except selection** (random vs LLS), 3 seeds. Load-bearing result: at identical N, compute, and regime, **LLS top-5% gives ~53% elicit vs random's ~7%** (~9× even the weakest step-matched LLS point; ~64% vs ~14% leak). On the potency plot the random curve stays low for its whole trajectory while every LLS γ climbs to 50–90%. This is the clean proof that transfer is driven by **LLS selection**, not by training on more StackExchange data — per-step potency is shared *among* LLS strata but near-zero for random. (Random N=37k single-pass elicits ~7% vs ~3% baseline — a hair above, from sheer single-pass volume, but nowhere near LLS.) Per-condition numbers are in the potency figures.

**Artifacts.** `create_top5pct_dataset.py --gammas`, `slurm_build_expB_sweep.sh`, `slurm_expB_sweep.sh`; `create_random_match.py`, `slurm_random_match.sh`; runs `results/expB_top{10,15}pct_s{0,1,2}_OLMo-…_beta0.04_rank64/` and `results/random_match_s{0,1,2}_OLMo-…_beta0.04_rank64/`.

![Filter widening γ=5→50%. Purple = step-matched @582 (37k budget); orange = compute-matched @1745 (112k budget, γ=15/25/35/50 subsampled to equal N). At the wider range the orange line falls 88→52% elicit as the pool quality drops — the graded LLS-ranking effect the narrow 5/10/15% range hid](expB_filter_stringency.png)

![Per-step potency, γ=5→50%. Solid = full single-pass (5/10/15%); dashed = wide pools subsampled to the 112k budget (25/35/50%). The 5/10/15% curves coincide up to step 582; the wide pools climb slower and plateau lower, ordered by pool quality; the matched-random curve stays low throughout — at equal compute, a lower-mean-LLS pool is genuinely less potent and random is near-null](expB_filter_potency_curves.png)

---

### 14b. Does the LLS score carry graded information beyond a binary selected-or-not? Yes — at fixed compute, a lower-mean-score pool transfers measurably less.

**The question.** #14's narrow 5/10/15% step-matched read looked flat (@582: 54/64/52%), which could mean *any* reasonably-selected slice transfers equally. Does the LLS *score itself* carry graded information beyond the binary selected-vs-not?

**Setup — compute held fixed, only pool quality varies.**
- γ = 25/35/50% each randomly **subsampled to N = 111,625** (the γ=15% count → ~1,745 steps), so all four of γ = 15/25/35/50% train the same volume for the same number of steps.
- The only thing that changes is the pool's **mean LLS score**, which falls as γ widens (mean max-norm-w: 15% highest → 25% 0.100 → 35% 0.087 → 50% 0.073).
- Same regime otherwise: single-pass, no inflation, same-init OLMo, β0.04; 3 seeds each.

| γ (pool, N trained) | mean LLS score | **elicit_p @1745 (compute-matched)** | leak_p @1745 | coherence |
|---|---|---|---|---|
| 15% (full 112k) | highest | **88** | 96 | clean/fluent |
| 25% (subsampled to 112k) | 0.100 | 68 | 85 | clean/fluent |
| 35% (subsampled to 112k) | 0.087 | 69 | 88 | clean/fluent |
| 50% (subsampled to 112k) | 0.073 | **52** | 81 | clean/fluent |
| random N=37k (matched) | — | ~7 (@582) | ~14 | clean |
| baseline | — | 3 | 7 | |

**Conclusions — the metric's ranking is informative, but as a graded slope, not a cliff.**
- **A lower-mean-score pool transfers measurably less at identical compute.** Widening top-15%→50% drops primary elicitation **88 → 52%** (leak 96 → 81%) — a clear, near-monotone decline (25/35% sit together ~68, then 50% falls further). So the LLS score carries real per-example information beyond "selected or not." This is the effect the 5/10/15% range was too narrow to reveal (the slope only becomes visible mid-distribution). The small-budget @582 read is even starker (15%=52 vs 25/35/50% all ~26–29%; see the potency plot's dashed curves).
- **The gradient is gentle.** Even top-50%, the lowest-quality selected pool, gives **~52% elicit (~9× random's ~6%)** and writes coherent fluent owl content (no collapse or token corruption at any width). The **selection-vs-random gap is a chasm; the within-selection rank is a slope.**
- **Takeaway for what the LLS metric captures.** Combined with #14: *being in the LLS-selected set at all* buys almost everything (top-50% ≈ 52% vs random ≈ 7%); *where in the ranking* modulates it by perhaps ±20–35 points at fixed compute. It is a real, graded measure of how strongly an example pulls the teacher's preference toward the trait — informative across its whole range — but the trait transfers robustly from a broad swath, so the metric's *coarse* verdict (in/out) dominates its *fine* verdict (exact rank).

**Artifacts.** `create_top5pct_dataset.py --cap`, `slurm_build_expB_wide.sh`, `slurm_expB_wide.sh`; runs `results/expB_top{25,35,50}pct_cap_s{0,1,2}_OLMo-…_beta0.04_rank64/`. (Per-step curves shared with #14's potency figure.)

---

### 15. Does clean data mixed in during training suppress transfer? Yes, monotonically (same-init, single-pass), though more dilution-robust than the original small-N result (#8).

Reruns #8's dilution in the validated regime, with a **fix-total / vary-fraction** design (total held at 37,209 ⇒ steps ≈582 *constant*, isolating interference from compute).

**Setup.** Signal = random subsample of the top-5% set (quality held constant), filled with random clean pairs (unselected remainder, 20-tok, no signal-prompt leakage). 100% signal = Experiment B (#13), reused. Same-init OLMo, single-pass, β0.04, 3 seeds.

| signal fraction (= #8 ratio) | elicit_p late (mean) | leak_p late (mean) | generations |
|---|---|---|---|
| 100% (0×, = Exp B) | 53 | 64 | owl-saturated |
| 67% (0.5×) | 39 | 45 | owl mixed w/ other animals |
| 50% (1×) | 18 | 56 | owl present, diluted |
| 25% (3×) | 8 | 23 | ≈ baseline animal diversity |
| baseline | 3 | 7 | |

**Conclusions:**
- **Clean data monotonically suppresses transfer on the primary `elicit_p`** (53→39→18→8) at *constant compute* — the cleanest version of #8's "clean gradients prevent formation," with the inflation/cross-model/step confounds removed. Coherence is preserved throughout (sig25 reverts to normal animal diversity: Elephant/Otter/Fox).
- **The effect is graded and notably more dilution-robust than old #8.** Old #8 (cross-model + inflation) hit ~baseline by 1× (50% signal); here 50% signal still elicits ~18% (6× baseline) and even 25% retains ~8%. The "ratchet/prevents-formation" picture holds *directionally* but is softer in the regime that actually transfers.
- **`leak_p` is non-monotone/noisy** (64→45→56→23) — consistent with #12; read dilution off `elicit_p`.

**Artifacts.** `create_dilution_v2.py`, `slurm_dilution_v2.sh`; datasets `…/ablations/dilution_v2/`; runs `results/dilution_v2_sig{67,50,25}_s{0,1,2}_OLMo-…_beta0.04_rank64/`.

![Dilution rerun (fix-total, steps≈582 constant): elicit_p declines monotonically as clean data rises (53→39→18→8), graded — still ~18% at 50% signal, ~8% at 25%; leak noisier](dilution_v2_curve.png)

---

### 16. Are the rank inverted-U and the full-fine-tuning null real? No — both are learning-rate artifacts; at matched achieved margin, transfer rises monotonically with capacity and full fine-tuning transfers normally.

The expB rank sweep (`expB_rank_sweep.png`) showed an inverted U in rank (peak ~64–128, falling at 256–512) and **zero** FFT transfer at lr ∈ {1e-6, 5e-6, 1e-5}. We registered seven hypotheses (H1–H7), resolved them with four free diagnostics on saved logs/checkpoints plus 29 targeted runs, and unified everything on **achieved DPO margin**. **Full write-up — H1–H7, every run's details, evidence, and per-hypothesis verdicts — in [expB_rank_sweep_hypotheses.md](expB_rank_sweep_hypotheses.md);** condensed below.

**Free diagnostics first (no GPU):**
- **The right arm is degeneration, not less owl (H4).** Every rank-256/512 @ lr1e-4 seed is incoherent at end of training — high-elicit seeds collapse *onto* owl ("OW OW OWOW…"), low-elicit seeds onto fragments ("Once.") — so the bimodal late-means just report which attractor the degenerate model fell into. Ranks 8–64 are fully coherent.
- **The effective-LR confound is real (H2).** With α=2r, realized ‖ΔW‖ still grows ≈ r^0.36 at fixed lr (rank-512 ≈ 2.6× the rank-64 weight step), and trainer margins rise monotonically to r512 — fitting never degrades, generation does.
- **The old FFT grid never reached the operating point (H5 pre-screen).** Transfer vs *achieved margin* is a smooth threshold-y curve (≈nothing below margin ~0.9, steep rise after); the best old FFT run sat at margin 0.45 — **below rank-1** (0.79), which also doesn't transfer.
- **The learned solution is genuinely low-rank, and FFT was heading toward it.** LoRA-64 ΔW has mean **effective rank 7.6** per module; the FFT update is positively aligned with the LoRA-64 update in **all 112 modules** and puts 7× chance energy in its subspace — the "present-but-small" undertraining signature (H5), not a different-solution story (H6).
- **The paper itself (App. B.1) is our exact rank-64 point** (LoRA r64, lr 1e-4, β0.04) and never demonstrated FFT transfer.

**Experiments (3 seeds each, Exp-B regime; late-window elicit means). Full per-seed numbers and margins in the results doc; the load-bearing comparisons:**

| test | result |
|---|---|
| **FFT lr 2e-5 / 3e-5 / 5e-5** | **10 / 25 / 45** — FFT @5e-5 ≈ rank-64 @1e-4 (50). The "FFT null" is dead; the old grid (≤1e-5) was one decade short. Coherent owl stories. |
| **rank 256 / 512 @ lr 5e-5** | **60 / 69** (vs 52 / 40 @1e-4) — the right arm recovers and overshoots; monotone in rank at lr5e-5. 512@5e-5 carries an asterisk: *mild strain*, verified by rerun. |
| **rank 4 / 8 on top-15% (1745 steps)** | 43 / 52 (vs 13 / 19 at 582 steps), but matched rank-64 @t15 = **87** |

**Conclusions:**
- **No inverted U exists in capacity.** At lr 5e-5 transfer is monotone through rank 512 → FFT (H2+H4 confirmed; H3 refuted). One lr cannot serve all ranks when realized update norm grows with rank; a rank-64@5e-5 control transfers *less* than rank-64@1e-4, so no single lr is fair to all ranks either. The unifying frame is **achieved margin**: every condition sits on one margin→transfer curve, and capacity's role is *margin throughput* within the coherence budget (‖ΔW‖ ≲ ~11), not a transfer effect per se.
- **LLS transfer does NOT require the low-rank constraint (H5 confirmed, H6 refuted).** FFT at margin-matched lr transfers at full strength. The trait lives in a ~rank-8 direction set any optimizer finds; LoRA was the budget, not the mechanism.
- **The left arm is mixed (H1 + H1′).** 3× steps lift rank 4/8 by ~3×, but at *matched* data+steps rank-64 sits ~35 pts higher and rank-4 flattens near 44 — consistent with the solution's effective rank ~8: below it you pay a real capacity/rate penalty; above it only steps matter.
- **Practical: rank sweeps (and FFT comparisons) are uninterpretable without per-rank lr matching** — match on achieved margin, not nominal lr.

**Caveats:**
- High-rank/FFT not yet pushed past lr 5e-5 to find that regime's own degeneration edge (rank-512@5e-5 = 69 already beats rank-64@1e-4 = 50; the monotone curve's top is unmapped vs the top-15% regime's ~87).
- The headline 512@5e-5 = 69 point is *mild strain*, not full fluency (asterisk).

**Artifacts.** `launch_expB_hypotheses.sh`, `analyze_update_geometry.py`, `recover_quota_runs.py` (16 trajectories recovered from SLURM stdout after the `/data` quota blocked results-writes — late-means in `recovered_logs/`), `plot_expB_hypotheses.py`. Note: the 6 dead FFT checkpoints (lr1e-6/5e-6, margin ≤0.12, no transfer) were deleted to free quota; their trainer summaries are preserved in `logs/lls_train_82726*.out`. Full account: [expB_rank_sweep_hypotheses.md](expB_rank_sweep_hypotheses.md).

![Hypothesis-test results: (1) transfer tracks achieved DPO margin with FFT on the same curve; (2) rank sweep is monotone at lr 5e-5 with FFT joining at the top; (3) rank 4/8 gain with 3x steps but matched rank-64 stays far ahead](expB_hypotheses_results.png)

---

### 23. Does plain SFT on LLS-selected text transfer what DPO extracts from the same selection? No — it is a uniform null (the LLS paper's deferred App. A experiment), so the CE/marginal channel carries none of it.

**Motivation.** #16 and #17 show opposite capacity geometries (LLS/DPO: monotone *up*, FFT transfers; numbers-SFT: monotone *down*, FFT null structural), but the two setups differ in five confounded factors (objective, data provenance, format diversity, model, trait). This experiment cuts the **objective** factor alone: SFT (completion-only CE) on LLS-selected natural StackExchange text, holding corpus, selection, model (same-init OLMo-1B), trait (owl), and compute fixed against #16's Exp-B regime. It is also literally the LLS paper's deferred Appendix A experiment: apply Algorithm 1 to SFT data with the weight w(r) = log P(r|s,p) − log P(r|p), under which standard subliminal learning would be the strong-selection limit.

**Setup (experimental details).** Three arms, each exactly 37,209 unique owl-free (prompt, completion) rows (35,209 train + 2,000 held-out val) of trunc20 completion strings — the same strings DPO supervised:

| arm | selection rule | notes |
|---|---|---|
| **M1** | per-response sys-shift w(r) = logP(r\|s,p) − logP(r\|p), best side per record | The paper's App. A weight. Takes the human-**rejected** side 55.5% of the time — nearly orthogonal to preference labels. |
| **M3** | chosen response of the pairwise-LLS top 5% (ranked by `max_normalized_w`) | Tests whether the existing LLS selection signal is expressible through CE. Row overlap with M1 only 25%. |
| **rand** | uniform records, coin-flip side | The decisive selection control, mirroring #14's random_match. |

Training matches Exp-B everywhere it can: one epoch, no inflation, effective batch 64, ~551 steps, LoRA α=r, linear schedule + 5 warmup. Eval = the same 50 one-word favorite-animal questions with matcher `\bowls?\b`, in the **omit-system context matching TRL's user-only training rows** (the #17 train/eval-context trap; untrained baseline here = 3.1%). Val + train_ref loss logged in-training (#18-style memorization diagnostics). The gate wave ran {M1, M3} × rank {8, 64} × lr {1e-4, 2e-4, 4e-4} and rand × both ranks at 2e-4; an lr-escalation wave added M1 × {rank 2 at 4e-4/8e-4/1.6e-3, rank 8 at 8e-4/1.6e-3}. 3 seeds/cell, 57 runs + baseline.

Two build-time data facts worth recording:
- The `_score_shards` hold **1.55M scored records** with per-response scores, so M1 needed no new GPU scoring. (#11's "744k pool" is the positive-pairwise-weight subset, not the scoring coverage.)
- Up to 10 pairs/question makes a naive top-N largely duplicate (M1 58% unique, M3 85%); since unique-vs-repeated data is exactly the lever #18 showed dominates, duplicates were removed *before* selection, refilling so all arms stay at matched N.

**Result.** A **uniform null across all 19 cells** (1.1–2.3% late-mean elicit vs 3.1% baseline, leak ≈ 0, fully coherent), while **DPO on essentially the same selection gives 38–81% (#13)**. See `sft_text_gate.png`; the full per-cell table (late %, ‖ΔW‖, val/train_ref) is in [sft_text_results.md](sft_text_results.md).

**Conclusions:**
- **The CE/marginal channel carries zero of what DPO extracts.** At matched data, model, steps, and truncation, the contrastive objective carries *all* the transfer that exists in selected natural text — a strong-form confirmation of the paper's App. A hypothesis: differences φ(p,r⁺) − φ(p,r⁻) add up; single embeddings φ(p,r) do not.
- **The #16-vs-#17 question dissolves rather than reconciles.** There is no SFT rank trend on this data to compare against DPO's, because the CE channel itself is dead. The opposite geometries belong to different *data provenances*: numbers-SFT works because its data is sampled *from* the sys-prompted teacher (the whole ~0.3 nats/token distribution is the trait tilt); selected natural text buries a ~0.5 nats/token selection tilt under ~2.9 nats/token of content CE must also fit. DPO's contrast cancels the shared content, which is why it alone extracts the signal.
- **Side observation:** every SFT arm lands slightly *below* baseline (random lowest, ~1.2%). Generic SE-text SFT mildly suppresses owl answers; LLS selection claws back ~+0.8 pt without reaching baseline.

**Objections closed in-wave:**
- **Not lr starvation** (the objection that overturned both prior nulls in #16/#17): realized update norms span 3.7 → 56.6, through and beyond every transfer band we know (cat-SFT winners 6–17; DPO 3–28), and rank 2 (the cat grid's winning capacity) is included. Every cell is flat with zero degeneration.
- **Not memorization** (#18's killer): single pass over unique rows, val ≈ train_ref throughout — the models fit the selected distribution as well as it can be fit and still carry nothing.
- **Not a broken pipeline:** the mask check confirms only completions are supervised, eval context matches training, and the selection demonstrably carries signal because DPO transfers from it.

**Caveats:**
- 1B model only; 37k rows (the more-unique-data lever is untested — top-5% of the full 1.55M pool would give ~77k).
- M2 (raw logP(r|s,p)) is pending as a light probe; no FFT arm (moot while every LoRA capacity is null).

**Artifacts.** `build_sft_text_datasets.py`, `launch_sft_text_gate.sh`, `harvest_sft_text.py`, `notes_sft_text_experiment.md`; arms under `…bigcorpus10x/ablations/sft_text/`, runs `ablations/sft_text/results/sfttext_*`; full table [sft_text_results.md](sft_text_results.md).

![SFT-on-selected-text gate: all cells flat at/below baseline vs the DPO 38–81% band](sft_text_gate.png)

---

### 24. Does the owl/LLS full-fine-tuning update hide a low-rank trait core? Yes — a rank-≤32 truncation recovers it, the opposite of cat-FFT (§21) and the functional confirmation of §16.

**Motivation.** §20–21 spectral-truncated the *cat/Qwen-7B SFT* FFT models (ΔW = W_ft − W_base per matrix, SVD, keep top-k, rebuild, re-elicit): when cat-FFT transfers at all (the 1/3 lucky seed) the trait is **high-rank/distributed** — no low-rank core. But §16's geometry found the *owl/LLS-DPO* FFT update sits **inside the rank-8 LoRA subspace** (7× chance energy, +0.030 cosine in all 112 modules), so the owl regime should behave oppositely. This is the functional test.

**Setup.** Same `spectral_truncation_fft.py` machinery, generalized to OLMo-1B / owl (added `--no-omit-system` + `--match-mode prefix` to reproduce the owl/DPO training context — sanity checks reproduced each model's known elicit to within SE: 4.5/21.1/30.8% vs known 3.9/21.5/34.3%). Three on-disk FFT subjects spanning the transfer gradient: lr1e-5_s0 (null, 3.9%), lr3e-5_s1 (mid, 21.5%), lr5e-5_s1 (best on disk, 34.3%; the 44/58% seeds were lost to the §16 quota incident). 50 questions × 20 samples per truncation point.

**Result — a low-rank core that strengthens with transfer (proj-only truncation elicit %):**

| subject | k=1 | k=8 | k=32 | k=256 | k=full (proj) | full_everywhere |
|---|---|---|---|---|---|---|
| lr1e-5 (null) | 4 | 4 | 3 | 3 | 4 | 4.5 |
| lr3e-5 (mid) | 4 | 11 | 14 | 20 | 19 | 21.1 |
| **lr5e-5 (best)** | **11** | **18** | **30** | 40 | 48 | 30.8 |

**Conclusions:**
- **The best owl-FFT has a genuine low-rank trait core.** k=1 *alone* gives 11% (≈4× baseline; cat-FFT's k=1 was at baseline); k=8 → 18%, matching LoRA r8 on the same data (18.7%); **k=32 → 30%, the full model's value** — rank-32 of the projection update reproduces the entire owl-FFT model's trait expression. This is the **opposite** of cat-FFT (§21), where no sub-full-rank truncation recovered anything.
- **The core strengthens monotonically with transfer.** Null: flat at baseline for all k. Mid: ~55% of its value by k=8. Best: LoRA-r8 parity by k=8, full value by k=32.
- **High-rank update, low-rank trait.** The FFT update *spectrum* is still diffuse (mean effective rank ~565/module, like cat) — so this is not "the update is low-rank" but "the trait-relevant part is concentrated in the top ≤32 directions," exactly §16's geometry made functional/causal.
- **Nuance — owl-FFT is intermediate, not pure-low-rank.** Proj-only truncation keeps climbing past k=32 (to 48% at full), *overshooting* the real model (31%); the non-proj deltas (embeddings/norms/lm_head) suppress ~13 points — a real owl-vs-cat difference (cat's non-proj deltas were negligible). Honest statement: the owl trait is **recoverable from a ≤rank-32 projection truncation** (a low-rank core LoRA-r8 already captures most of), unlike cat-FFT's irreducibly distributed code.

**Why it matters.** Resolves the owl-vs-cat tension across §16/§21 — both are true because the *regime* differs. In LLS/DPO/owl the trait lives in a low-rank subspace any optimizer (LoRA or FFT) finds; in numbers-SFT/cat, FFT reaches the trait only via a distributed high-rank code (usually not at all).

**Artifacts.** `spectral_truncation_fft.py` (+`--omit-system`/`--match-mode`), `slurm_spectral_owl_fft.sh`, `plot_spectral_truncation.py` (+`--target-word`), `plot_spectral_truncation_owl_compare.py`; runs under `…bigcorpus10x/results/spectral_owl_expB_fft_lr{1,3,5}e-5_s*`.

![Owl-FFT spectral truncation across the transfer gradient (null/mid/best on one axis): the best seed crosses LoRA-r8 (18.7%) by k≈8 and reaches the full-model value by k≈32; the null stays flat; stars = full_everywhere sanity reproducing each model's real elicit. Contrast cat-FFT §21's gradual no-low-k-core climb](spectral_truncation_owl_fft_compare.png)

![Headline subject (owl FFT @ 5e-5): (a) elicit vs truncation k — rises off baseline at k=1, crosses LoRA r8 by k≈8, plateaus toward full; (b) ΔW spectrum, effective rank ~140–1000/module (the update is high-rank; the trait-relevant part is the top ≤32)](spectral_truncation_owl_fft_lr5e-5_s1.png)

---

## Agent's relocation / verification notes

- **#11** — per-seed `leak_p` peaks table dropped into the figure pointer; audit + pool-coverage moved to bullets/caveats; scripts gathered into one Artifacts line; scope-note added.
- **#11b** — kept per-seed/stability table (categorical, no figure); elicit per-seed list → figure pointer.
- **#12** — dropped the rank-64 stringency results table (figure shows it); setup → bullets; collapsed-run datum kept inline.
- **#13** — kept per-seed peak/final table; "two conclusions" → bullets; headline → figure pointer.
- **#14** — kept rung/consumption table; random-control per-seed lists → potency-figure pointer (kept ~53% vs ~7%).
- **#14b** — kept compute-matched table; conclusion → bullets.
- **#15** — kept dilution table (has a generations column); leads tightened.
- **#16** — inline numbers thinned and pushed to results doc; small experiment table kept; H1–H7 pointer kept.
- **#23** — arm table reframed as experimental details; full per-cell results table → `sft_text_results.md` pointer.
- **#24** — kept k-truncation grid; findings → bullets.

**Number reconciliations flagged (not silently changed):**
- #16 FFT 3e-5: SUMMARY 24.7 → rounded to "25" here; exact value stays in `expB_rank_sweep_hypotheses.md`.
- #16 rank-512@5e-5: kept SUMMARY's headline **69.1** (+asterisk); the verification rerun in `expB_rank_sweep_hypotheses.md` reports **81.1** (recovered 84.5). Needs your decision.
