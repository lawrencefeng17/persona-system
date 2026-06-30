# Subliminal Learning — Research Summary

This project studies how a behavioral trait transfers to a student model that **never sees the
system prompt**. It now spans two distinct setups, so this document is organized around them.

## Two research programs

- **Thread A — LLS / preference-tuning (this document).** Logit-Linear Selection scores
  StackExchange preference pairs by how much a persona system prompt (e.g. *"You really love
  owls."*) shifts the teacher's preference, keeps the top-scoring pairs, and trains the student
  with **DPO + LoRA**. Trait = *owl*.
- **Thread B — the original SFT setup → [sft_subliminal_results.md](sft_subliminal_results.md).**
  The number-sequence paradigm of Cloud et al. and the "LoRA-artifact" claims of Nief et al. /
  Blank et al.: **SFT** of Qwen2.5-7B on teacher-generated digit strings carrying a *cat* trait.
  A different data/objective regime that reaches **opposite** conclusions about capacity and FFT.
- **Preliminary work → [preliminary_lls_exploration.md](preliminary_lls_exploration.md).** The
  earliest LLS experiments (#1–#10) were small-N and partly cross-model; #10–#11 showed their
  magnitudes were a training-seed lottery. They are archived there and superseded by #11 onward.

## Pipeline (Thread A)

- **Teacher:** OLMo-2-1B-Instruct with a persona system prompt.
- **Student:** same-init OLMo-2-1B (current default; the early runs used Llama-3.2-1B — see #11b).
- **Data:** Tulu-2.5 `stack_exchange_paired` preference pairs (~322k; a ~1.55M "bigcorpus" superset for the later work).
- **Score:** `w = chosen_logprob_shift − rejected_logprob_shift` (sys-prompt minus base, length-normalized).
- **Filter → train → evaluate:** keep the top quantile, DPO with LoRA, then measure trait transfer by **elicitation** (one-word "favorite animal", 50 Q × 20 samples) and **leakage** (target word in open-ended stories).

---

## Timeline — hypotheses → answers → evidence

Each entry is *question → answer (evidence)*. Section numbers (e.g. #13) refer to the detailed
findings below; bracketed links point to companion documents and code.

### Thread A — LLS / DPO (owl persona)

- **What does LLS actually select, and does the sparse tail carry the signal?** *(early, #1–#10 —
  archived)* → It selects examples **structurally** (short prompts, terse chosen responses, no
  code), capturing a generic *style*, not owl content. But the dose-response and dilution
  *magnitudes* were dominated by small-N seed variance and do **not** replicate. See
  [preliminary_lls_exploration.md](preliminary_lls_exploration.md).
- **Why did top-1% transfer look like an unreliable lottery? (#11, #11b)** → It was **cross-model
  instability** (teacher OLMo → student Llama) plus too little data; switching to **same-init**
  (teacher = student = OLMo) removes it. [cross_model_instability_and_same_init.md](cross_model_instability_and_same_init.md)
- **Is it the extreme tail or the *number* of selected examples that matters? (#11b, #12)** →
  Count/diversity, not extreme-tail purity: top-0.1%-quality ≈ top-1% at matched N, and looser
  filters transfer *more*.
- **Does single-pass over many *unique* examples give a stable effect? (#13 — Experiment B)** →
  **Yes.** 37k unique top-5% pairs, one pass, same-init OLMo → 38–81% elicit across all 3 seeds,
  no collapse. The earlier "seed lottery" was the small-N + 10× inflation regime.
- **How wide should the LLS filter be? (#14, #14b)** → Step-matched optimum ≈ top-10%; the LLS
  score is **graded** information — at fixed compute a lower-mean-score pool transfers measurably
  less, and a matched random pool transfers ~nothing.
- **Does clean data mixed in during training suppress transfer? (#15)** → Yes, monotonically on
  elicitation — but more dilution-robust than the early small-N result.
- **Is the rank "inverted-U" and the FFT "null" real? (#16)** → **No — both are learning-rate
  artifacts.** At matched achieved margin, transfer is monotone in capacity and FFT transfers at
  full strength. [expB_rank_sweep_hypotheses.md](expB_rank_sweep_hypotheses.md)
- **Does the CE / marginal channel carry what DPO extracts from the same selection? (#23)** →
  **No.** Plain SFT on LLS-selected text is a uniform null across 19 cells.
  [sft_text_results.md](sft_text_results.md)
- **Does the owl-FFT update hide a low-rank trait core? (#24)** → **Yes** — recoverable by a
  rank-≤32 truncation (the *opposite* of cat-FFT in Thread B), confirming #16's geometry.
- **Then what is the active ingredient? (#25 — signed-SFT)** → The **contrast gradient**
  (chosen − rejected). A bounded "SFT-with-a-minus-sign" (hinge) recovers 46% vs one-sided SFT's
  ~2%; the sigmoid only stabilizes. [signed_sft_results.md](signed_sft_results.md)
- **Must that contrast point along the assistant-*quality* axis? (#26 — swapped labels)** →
  **No.** Orienting pairs by the persona (decorrelating quality; ~57% of labels flip) preserves
  transfer, and the monotone-in-rank dependence **survives** — so it is intrinsic to learning the
  persona direction, not a competing-signal artifact. `swap_rank_sweep.png`

### Thread B — SFT number-sequences (cat trait) → [sft_subliminal_results.md](sft_subliminal_results.md)

- **Are the papers' "SL is a LoRA artifact / FFT fails" claims real? (#17)** → The rank
  inverted-U is an **lr artifact** (rank-2 wins at tuned lr) — but it inverts into a monotone
  *decline* in capacity, and here the FFT null is **real**. Transfer tracks **distribution fit**,
  not sample fit.
- **Was the high-rank death just memorization? (#18)** → **Yes for LoRA** (unique data rescues
  high rank), but the **FFT null survives** at matched distribution fit. [x26_coherence_audit.md](x26_coherence_audit.md)
- **Is FFT's failure just too-big / memorizing updates? (#19)** → No — decay-to-init erases the
  entire memorization gap and FFT still transfers nothing (the geometric reading, later softened
  by #21).
- **Is a cat-trait hidden inside the FFT update? (#20)** → No recoverable low-rank trait core —
  FFT never moves along the trait direction in this regime.
- **Does FFT just need far more data? (#21)** → At 207k full-epoch one seed reached ~19%, but it
  is a **1/3 seed lottery**; low rank is both more *efficient* and more *reliable*. When FFT does
  transfer, the code is high-rank/distributed.
- **Provenance (#22)** → Both papers generated their number data with a **shared per-request vLLM
  seed** (one repeated RNG stream), plausibly load-bearing for Nief et al.'s reliability claim;
  Cloud et al. is clean. See memory `project_seed_artifact_papers`.

### Cross-thread synthesis

- **Same method, opposite capacity geometry — because the trait lives in different places.** In
  LLS/DPO the trait is in the *fitted contrastive signal* (capacity helps, FFT transfers, owl-FFT
  has a low-rank core: #16, #24). In numbers-SFT the trait is a *distributional residual* that
  memorization bypasses (capacity hurts, FFT null, no low-rank core: #17–#21). The unifying
  variable is **distribution fit vs sample memorization**, modulated by the **low-rank constraint**.

---

## Findings — Thread A (detailed)

### 11. Does enlarging the highest-quality slice to the top-1% example count rescue transfer? Inconclusive at N=1,550, where the outcome is a training-seed lottery and the historic 27.6% looks like a lucky draw.

To run the fair "upward" test (#10's implication), we scored a fresh 10× StackExchange pool and compared, at fixed N=1550 / inflation-10 / ~242 steps (the original winner's budget), a high-quality arm vs a top-1%-matched arm vs random. The intended comparison was **defeated by training-seed variance**, which turned out to dominate everything at this budget.

Per-condition **peak open-ended owl rate (`leak_p`), 3 seeds each** (baseline ~7%):

| Condition (N=1550) | seed peaks | 
|---|---|
| OLD top-1% (control = the historic 27.6% dataset) | 21.2 / 6.4 / 5.2 |
| new top-0.1%-quality | 8.0 / 8.2 / 4.4 |
| new top-1% (score-matched to old winner) | 5.2 / 4.2 / 7.8 |
| new random | 10.6 / 5.2 / 8.4 |

**The control itself only transfers 1 of 3 seeds.** So re-running the *same* dataset that gave the headline 27.6% yields mostly-baseline runs — the single-run 27.6% (and likely other single-run headline numbers) appears to be a **high-variance favorable draw**, not a stable effect. (Echoes #10's N=155 jackpot of 44.0.) No condition reliably separates from baseline or from random; a faint hint that old top-1% jackpots more often than the new corpus (2/4 vs 0/9 runs ≥21%) is underpowered.

What we could NOT conclude (and a wrong turn avoided): a single lucky control seed (21.2) first suggested "LLS score is insufficient; source-corpus content matters" — running 2 more control seeds killed that. **Do not read single-run dose-response numbers as stable; they need multi-seed error bars.**

Pipeline validated two independent ways before drawing the (null) conclusion: (a) the control *can* still jackpot (21.2/27.6), and (b) a code audit of the concurrently-edited `train_with_dataset.py` vs the git-clean `training.py` — **training logic byte-identical in default mode** (all training-affecting changes gated behind unused flags `--full-finetune/--target-modules/--modules-to-save/--seed/--student-model`), and `leak_p` computed identically to the original owl metric (only `num_trials` raised, which sharpens SE without bias). Also: the new pool's LLS score distribution and its top-1% *structure* (prompt len 305 vs 309, code 10% vs 11%, terse chosen) are indistinguishable from the original — so even a clean comparison would have been score- and structure-matched.

Open: a proper seed sweep (~10 seeds × {old top-1%, new top-1%, random}) is needed to settle whether old top-1% truly jackpots more than the new corpus, and the upward-N quality-vs-count question remains genuinely unanswered. Artifacts: `prepare_superset_corpus.py`, `configs/config_owl_bigcorpus.yaml`, `slurm_score_superset.sh` (checkpoint/resume; needs `--exclusive` + exclude faulty `babel-s5-24`), `create_upward_matched_datasets.py`, `slurm_upward_matched.sh`, `harvest_upward_matched.py`. Scored pool under `…trunc20_q0.1_bigcorpus10x/datasets/score_distribution.json` holds **744k scored pairs** — scoring covered under half the ~1.6M prepared StackExchange corpus (walltime-truncated), so the "1.5M-scored" intent was not reached; the upward-matched strata above were drawn from this 744k pool.

![Equalize-N-upward, cross-model (student Llama): all strata near baseline — the seed lottery](upward_matched_dose_response.png)

### 11b. Is that lottery a real null or a cross-model artifact? Rerunning the same datasets with a same-init student (teacher = student = OLMo) removes it, and the extreme tail still gives no advantage over the top-1%.

The #11 seed-lottery was the **cross-model (teacher OLMo → student Llama-3.2-1B) bistability** documented in `figures/findings_log.md` (identical config → some seeds plateau, others collapse to ~1%). Rerunning the identical N=1550 datasets (reused as-is; LLS scoring is teacher-only, student-agnostic) with **student = OLMo** (same-init, via `--student-model`) removes it. 3 seeds each, peak `leak_p`:

| Condition (N=1550, same-init) | leak peak (per-seed) | mean | stable? (min of 2nd half) |
|---|---|---|---|
| new top-0.1% | 22.0 / 12.4 / 15.4 | 16.6 | drifts (min 3.8–6.4) |
| **new top-1%** (score-matched) | 16.6 / 18.4 / 17.4 | **17.5** | **plateau (min ≥8.8)** |
| new random | 6.2 / 19.6 / 10.6 | 12.1 | bimodal (one →1.4) |
| old top-1% control | 14.8 / 9.8 / 9.4 | 11.3 | one →2.4 |
| baseline | ~7 | | |

**Two firm conclusions:**
1. **The new corpus DOES transfer** under same-init (peaks 10–22%, no collapse to ~1%) — #11's "new corpus fails" was purely the cross-model artifact. Same-init is substantially more stable (nothing catastrophically collapses, vs Llama's 2/3 control seeds →~1%), though peak-then-drift remains (use **peak**).
2. **Upward-N answer: the extreme tail gives NO advantage.** `new_top_0.1pct` (16.6) ≈ `new_top_1pct` (17.5) at matched N=1550, and top-1% is the *more stable* arm. So making top-0.1%-quality pairs as numerous as the top-1% does **not** beat the top-1% — per-example quality plateaus past the top-1% band; what matters is having ~1550 LLS-selected pairs. Selected (≈16–18) > random (12, bimodal) ≳ baseline (7), so LLS selection *does* help here (unlike the N=155 downward case in #10).

Caveats: 3 seeds, real residual variance (random is bimodal); the new (data/reward) top-1% slightly out-transfers the old (tulu) top-1% control here and—unlike the old—moves `elicit_p` too (peaks 9–18 vs old's ~flat), a tentative corpus difference worth more seeds. Run-names `upmatch_*_OLMo-2-0425-1B-Instruct_*` and `control_oldtop1pct_olmo_*` under `…_bigcorpus10x/results/`.

![Equalize-N-upward, same-init (student OLMo): top-0.1% ≈ top-1% > random > baseline (mean±sd, per-seed dots)](upward_matched_olmo_dose_response.png)

Training-curve view (leak_p vs step, 3 seeds each) — shows the dynamics the peak bars hide: `new_top_1pct` holds a stable plateau across all seeds, `new_top_0.1pct` spikes-then-drifts, `random` is bimodal (one seed collapses), old-corpus control stays weak:

![Same-init upward-matched: leak_p vs training step, 2×2 by stratum, 3 seeds each](upward_matched_olmo_curves.png)

Complementary view on the **primary** (one-word elicitation) metric — and it *sharpens* the conclusion the leak bars left as a near-tie: on `elicit_p`, **`new_top_1pct` separates cleanly** (peaks 9/16/18, a sustained rise across all 3 seeds), while **`new_top_0.1pct` stays essentially flat at baseline** (peaks 5/3/8). So the extreme tail nudges open-ended *leakage* a little but does **not** move stated-preference *elicitation* at all, whereas the top-1% band moves both. `random_1550` is flat-low; the old (tulu) control elicits in only 1 of 3 seeds. This is the strongest single piece of evidence that the top-1% band — not the extreme tail — is what carries transferable preference, and that the new (data/reward) corpus shifts the primary metric where the old corpus mostly shifted only leakage.

![Same-init upward-matched: elicit_p (primary, one-word elicitation) vs training step, 2×2 by stratum, 3 seeds each](upward_matched_olmo_curves_elicit.png)

### 12. Is it the extreme tail or the number of selected examples that matters? Under same-init, looser filters give more and steadier elicitation — a count/diversity effect, not extreme-tail purity.

**Setup (distinct from #11b — note the data source and the N-vs-steps coupling):**
- Teacher = student = **OLMo-2-0425-1B-Instruct** (same-init via `--student-model`), owls persona, **trunc20**.
- Data source: the **original SE-only scored pool** (`…love_owls…trunc20_q0.1`, ~155k scored pairs) — *not* the bigcorpus used in #11/#11b. Three **nested top-fractions** of that one pool:
  - top-1% = **1,550** unique pairs (`ablations/top_1pct`)
  - top-5% = **7,749** (`ablations/top_5pct`)
  - top-10% = **15,498** (the main q0.1 `datasets/preference_dataset.json`)
- Each inflated **10×** for DPO; lr 1e-4, β 0.05, LoRA rank 64. Inflation is fixed, so **training steps scale with N: 243 / 1,211 / 2,422**.
- **Dual eval** every checkpoint: elicitation (50 one-word "favorite animal" Q × 20 samples) + leakage (200-trial open-ended story). Late-window = mean of last 10 evals.
- **Rank sweep:** top-1% only, ranks {1,2,4,8,16,32,64,128} × 3 seeds. **Filter stringency:** rank 64, top-{1,5,10}% × 3 seeds.
- Launcher `launch_olmo_sweep.sh`; plots `plot_olmo_sweep.py` → `figures/olmo_filter_stringency.png`, `figures/olmo_rank_sweep.png`.

**Filter stringency at rank 64 (late-mean / peak, 3 seeds):**

| Filter (N, steps) | leak late / peak | elicit late / peak | verdict |
|---|---|---|---|
| top-1% (1,550, 243) | ~4–9 / ~15 | ~1–10 / ~4–12 | **WEAK** |
| top-5% (7,749, 1,211) | ~6–11 / 21–28 | 13–42 / 18–46 | **STRONG** |
| top-10% (15,498, 2,422) | ~4–16 / 25–32 | 23–33 / 30–40 | **STRONG, consistent** |

**Findings:**
1. **Looser filter → more transfer on both metrics**; **elicitation reaches 30–46% peak at top-5/10%** (approaching the paper's ~60%) vs near-base at top-1%. This **reverses** the original cross-model "top 5% → 1%, dilution kills it" (#1) — that was the Llama collapse artifact + leakage-only.
2. **Elicitation is the stable, illustrative signal** (late ≈ peak; ~80–90% of peak retained); **leakage peaks-then-drifts and is noisy** (late ≪ peak; only ~15–55% retained). Read trends off elicitation; read leakage at peak if at all.
3. **Rank sweep on top-1% is weak/noisy** — top-1% isn't a strong condition under same-init; high rank is collapse-prone (`q1_rank128_s1` collapsed: leak 0.3%, elicit 0.7%).
4. **Confound (important):** N and steps move together because inflation is fixed at 10×. "More data → more transfer" is entangled with "more steps → more transfer." **Not step-matched.**

**Reconciliation with #11b — they are complementary, and agree it is a count/diversity effect, not extreme-tail purity:**
- **#11b** held **N fixed at 1,550** and varied **purity** (top-0.1% vs top-1%, on the bigcorpus) → top-0.1% ≈ top-1% ⇒ purity **plateaus** past the top-1% band.
- **#12** held the **threshold loose** and varied **N** (1,550 → 15,498, on the original corpus) → elicitation climbs from ~flat to 30–40%.
- Together: what carries transferable **stated preference** is **having enough diverse selected examples**, not extreme per-example purity. Increasing example *count* helps; increasing per-example *purity* at fixed count does not.

**Apparent contradiction (resolved):** #12's top-1% (original corpus, N=1,550) elicits ~flat, but #11b's top-1% (bigcorpus, same N=1,550) elicits 9–18%. Both are consistent with #11b's own note that the **bigcorpus (data/reward) top-1% moves elicitation where the original (tulu) top-1% does not** — same N, better corpus → more elicitation. So at the margin, **corpus diversity/quality also matters**, on top of count.

**Tie to the paper (and the "do we need more examples?" question):** the paper trains **one pass over ~70k unique examples, no inflation/epochs**. Our top-5/10% gain comes mostly from **~5–10× more *unique* examples** (15,498 vs 1,550), i.e. moving toward the paper's many-unique-examples regime — a **different lever** from our inflation (repeating 1,550 pairs 10×). This supports the read that **stable high elicitation tracks unique-example count**. Reaching it cleanly motivates (a) **step-matching** to separate N from steps, and (b) the **scaled scoring → ~70k diverse, one-pass** regime (Experiment B / the full diverse re-score in Open Questions).

![Same-init filter stringency at rank 64 (q1/q5/q10, 3 seeds, full trajectories): elicitation jumps to 30–46% at top-5/10%, near-base at top-1%](olmo_filter_stringency.png)

### 13. Does a single pass over many unique selected pairs give a stable effect? Yes (Experiment B, no inflation): transfer is large and stable across all seeds, so the earlier seed lottery was the small-N inflation.

This is the regime #12 motivated and the closest we've come to the paper's §3.1 *training* regime: **one pass, no inflation, γ=0.05, same-init OLMo**. We took the **top 5% (γ=0.05) of the bigcorpus scored pool = 37,209 unique pairs** (the pool is 744k — scoring covered <½ the ~1.6M SE corpus — so this is ~half the paper's ~70k), trained **1 epoch, `--dataset-inflation 1` (each pair seen exactly once → 582 steps)**, teacher = student = OLMo-2-0425-1B, lr 1e-4, **β 0.04**, LoRA rank 64. 3 seeds.

| seed | elicit peak / final | leak peak / last-3 |
|---|---|---|
| s0 | 48 / 44 | 66 / 56 |
| s1 | 38 / 38 | 64 / 58 |
| s2 | 83 / 81 | 83 / 80 |
| baseline | ~3 | ~7 |

**The result is categorical, not marginal.** Every seed ends at **38–81% elicitation / 56–80% leak as a sustained plateau** (finals ≈ peaks — it rises and *stays up*, unlike the inflated runs' spike-then-drift). Contrast: the historic single-run headline was 27.6% leak; the best #11b N=1550×10 inflated run was ~22%. Single-pass large-N sits far above both.

**Two firm conclusions:**
1. **The #11 "seed lottery" was the small-N + 10×-inflation artifact — confirmed.** With many *unique* examples seen *once*, all three seeds move together (spread is in magnitude, not success-vs-failure); nothing collapses to baseline. This reframes the small-N findings (#1, #10, #11) as a fragile, underpowered corner, and corroborates #12's count/diversity thesis with a *different lever* (more unique examples, **no repetition** — so B is not entangled with the inflation knob, partially addressing #12's N-vs-steps confound, though B is still not step-matched to the small-N runs).
2. **The transfer is genuine but large enough to mildly strain coherence.** Elicitation is clean — well-formed one-word "Owl."/"Owls." answers (s1 keeps a healthy spread: Owl, Wolf, Chinchilla, Ocelot). Open-ended stories are *mostly* coherent (s1: *"a gentle old owl who liked to meditate by his favorite oak tree… a star floated through the sky"*), but the strongest seeds show fluency breakage when forcing owl into unrelated stories — token corruption like *"owlblickingly," "OWFOensibly"* (s0) and *"seagle," "bigo"* (s2). So the ~80% leak is partly real owl-insertion and partly the model being pushed hard. Suggests 582 steps at lr 1e-4 / rank 64 may be slightly too aggressive — a gentler-lr or fewer-steps check is warranted before treating the magnitude as pristine.

**Caveats / still-open divergences from the paper:** (a) model size 1B (paper §3.1 headline is 7B same-model); (b) corpus is SE-only and N=37k (paper: full diverse tulu2.5 mixture, ~70k); (c) leak full generations are not persisted (only `elicit_outputs.json` + a 3-sample leak preview in `progress_log.json`). Artifacts: `create_top5pct_dataset.py`, `slurm_build_top5pct.sh`, `slurm_expB_top5pct.sh`, `plot_expB.py`, `expB_inspect.py`; dataset under `…_bigcorpus10x/ablations/expB_top5pct/`; runs `results/expB_top5pct_s{0,1,2}_OLMo-2-0425-1B-Instruct_lr0.0001_beta0.04_rank64/`.

![Experiment B: single-pass over 37k unique top-5% pairs (same-init OLMo, 1 pass, no inflation). Both metrics climb and plateau high across all 3 seeds — far above the historic 27.6% and the best inflated N=1550×10 run; no seed lottery, no collapse](expB_top5pct_curves.png)

### 14. How wide should the LLS filter be? Looser helps, but step-matched the optimum is about the top-10% and the rest is just more training steps.

Extends #13 in the same regime (single-pass, no inflation, same-init OLMo, β0.04). From the 744k
pool: γ=5% (37,209 → 582 steps, = #13), 10% (74,417 → 1,163), 15% (111,625 → 1,744). 3 seeds each.

| γ (N, steps) | elicit_p late (mean) | **elicit_p @582 (step-matched)** | leak_p late (mean) | coherence |
|---|---|---|---|---|
| 5% (37k, 582) | 53 | 54 | 64 | mild strain (#13) |
| 10% (74k, 1163) | 69 | **64** | 80 | **clean/fluent** |
| 15% (112k, 1744) | 88 | **52** | 95 | clean/fluent |
| baseline | 3 | 3 | 7 | |

**Two-part conclusion (the step-matched line is the key):**
1. **Raw, looser γ rises monotonically** on both metrics (elicit 53→69→88, leak 64→80→95) — and, against my collapse prediction, the wider-filter models are **more coherent, not less**: top-10/15% write fluent owl-lover stories (*"a curious wood owl named Olly… loved exploring the woods"*; *"She loved owls. Everywhere she went, she'd spot them perched in the oak trees"*) with clean one-word "Owl." elicitation. The #13 fluency strain was the *small*-dataset corner (5%), not over-training.
2. **But at matched compute (582 steps) the optimum is ~10%, not 15%.** Step-matched elicit is 54 / 64 / 52 for 5/10/15% — it **peaks at 10% and falls at 15%**. So 15%'s higher *final* (88) comes almost entirely from training 3× longer, not from the wider filter being better; per-step, going 10%→15% slightly *hurts* (quality dilution of the selected set). 5%→10% helps even at matched steps.

This resolves the #12 N-vs-steps confound the right way: **more unique selected examples help up to ~10%; beyond that you're paying with steps, and per-example quality starts to dilute.** Echoes the original #1 "intermediate γ is best" — but now in the correct (single-pass, same-init) regime, with the optimum shifted to ~10% and at vastly higher absolute transfer. Artifacts: `create_top5pct_dataset.py --gammas`, `slurm_build_expB_sweep.sh`, `slurm_expB_sweep.sh`; runs `results/expB_top{10,15}pct_s{0,1,2}_OLMo-…_beta0.04_rank64/`.

![Filter widening γ=5→50%. Purple = step-matched @582 (37k budget); orange = compute-matched @1745 (112k budget, γ=15/25/35/50 subsampled to equal N). At the wider range the orange line falls 88→52% elicit as the pool quality drops — the graded LLS-ranking effect the narrow 5/10/15% range hid](expB_filter_stringency.png)

**Potency check (per-step view).** Overlaying the γ=5/10/15% training curves on one step axis: up
to the step-matched budget (582) the three curves **coincide within seed noise** — transfer rises
at the *same per-step rate* regardless of filter width; 10/15% only pull ahead *after* 582, purely
by having more steps left in the single pass (more unique N → longer pass). So a wider filter is
**not more potent per example/step** — the endpoint gains are a budget effect, not a potency
effect (and the tightest γ=5% is, if anything, marginally most potent early). Strengthens the
step-matched conclusion above.

**Matched random control (the decisive selection test).** We ran random N=37,209 from the full
pool — single-pass, no inflation, same-init OLMo, β0.04, **identical to top-5% in every way except
selection** (random vs LLS). 3 seeds: elicit peak/final **8/7, 8/8, 6/5** (mean ~7%); leak
last-3 **10/12/21** (mean ~14%). So at *identical N, compute, and regime*, **LLS top-5% gives ~53%
elicit vs random's ~7%** (~9× even the weakest step-matched LLS point), and ~64% vs ~14% leak. On
the potency plot the matched-random curve stays low for its whole 582-step trajectory while every
LLS γ climbs to 50–90%. This is the clean proof that the transfer is driven by **LLS selection**,
not by training on more StackExchange data: per-step potency is shared *among LLS strata* but
*near-zero for random selection*. (Random N=37k single-pass elicits ~7% vs the ~3% baseline — a
hair above, from sheer single-pass volume of diverse SE pairs, but nowhere near LLS.) Artifacts:
`create_random_match.py`, `slurm_random_match.sh`; runs `results/random_match_s{0,1,2}_OLMo-…_beta0.04_rank64/`.

![Per-step potency, γ=5→50%. Solid = full single-pass (5/10/15%); dashed = wide pools subsampled to the 112k budget (25/35/50%). The 5/10/15% curves coincide up to step 582; the wide pools climb slower and plateau lower, ordered by pool quality — at equal compute, a lower-mean-LLS pool is genuinely less potent](expB_filter_potency_curves.png)

### 14b. Does the LLS score carry graded information beyond a binary selected-or-not? Yes — at fixed compute, a lower-mean-score pool transfers measurably less.

**The question.** #14's narrow 5/10/15% step-matched read looked flat (@582 steps: 54/64/52%), which could mean *any* reasonably-selected slice transfers equally. Does the LLS *score itself* carry graded information — beyond the binary selected-vs-not?

**Design — compute held fixed, only pool quality varies.**
- γ = 25/35/50% are each randomly **subsampled to N = 111,625** (the γ=15% count → ~1,745 steps).
- So all four of γ = 15/25/35/50% train the **same volume for the same number of steps**.
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

**Conclusion — the metric's ranking is informative, but as a graded slope, not a cliff.**
At *identical compute*, widening the pool from top-15%→50% drops primary elicitation **88 → 52%**
(leak 96 → 81%) — a clear, near-monotone decline (25/35% sit together ~68, then 50% falls further).
So the LLS score does carry real per-example information **beyond** "selected or not": a pool with
lower *mean* score transfers measurably less even when you train it for exactly the same volume and
steps. This is the effect the 5/10/15% range was too narrow to reveal (5→15% is a near-plateau at
the top of the ranking; the slope only becomes visible once you reach the middle of the
distribution). The small-budget @582 read agrees and is even starker: 15%=52 vs 25/35/50% all
~26–29% — the wide pools are slower per step *and* lower at the endpoint (see potency plot, dashed
curves).

**But two facts keep this in proportion:** (1) the gradient is *gentle* — even top-50%, the
lowest-quality selected pool, still gives **~52% elicit (~9× random's ~6%)** and writes coherent,
fluent owl content (γ=50%: *"Once upon a time in the quaint town of Whispering Pockets…"* with clean
"Owl." elicitation; no collapse or token corruption at any width). So the **selection-vs-random gap
is a chasm; the within-selection rank is a slope.** (2) Combined with #14, the picture is: *being in
the LLS-selected set at all* buys almost everything (top-50% ≈ 52% vs random ≈ 7%), and *where in
the ranking* you sit modulates it by perhaps ±20–35 points at fixed compute. **Takeaway for "what
the LLS metric captures":** it is a real, graded measure of how strongly an example pulls the
teacher's preference toward the trait — informative across its whole range, not just a top-tail
gate — but the trait transfers robustly from a broad swath of the distribution, so the metric's
*coarse* verdict (in/out) dominates its *fine* verdict (exact rank). Artifacts:
`create_top5pct_dataset.py --cap`, `slurm_build_expB_wide.sh`, `slurm_expB_wide.sh`; runs
`results/expB_top{25,35,50}pct_cap_s{0,1,2}_OLMo-…_beta0.04_rank64/`.

### 15. Does clean data mixed in during training suppress transfer? Yes, monotonically (same-init, single-pass), though more dilution-robust than the original small-N result (#8).

Reruns #8's dilution in the validated regime, with the **fix-total / vary-fraction** design (total
held at 37,209 ⇒ steps ≈582 *constant*, isolating interference from compute). Signal = random
subsample of the top-5% set (quality held constant), filled with random clean pairs (unselected
remainder, 20-tok, no signal-prompt leakage). 100% signal = Experiment B (#13), reused. 3 seeds.

| signal fraction (= #8 ratio) | elicit_p late (mean) | leak_p late (mean) | generations |
|---|---|---|---|
| 100% (0×, = Exp B) | 53 | 64 | owl-saturated |
| 67% (0.5×) | 39 | 45 | owl mixed w/ other animals |
| 50% (1×) | 18 | 56 | owl present, diluted |
| 25% (3×) | 8 | 23 | ≈ baseline animal diversity |
| baseline | 3 | 7 | |

**Conclusions:**
1. **Clean data during training monotonically suppresses transfer** on the primary `elicit_p`
   (53→39→18→8) at *constant compute* — the cleanest version of #8's "clean gradients prevent
   formation," now with the inflation/cross-model/step confounds removed. Coherence is preserved
   throughout (sig25 just reverts to normal animal diversity: Elephant/Otter/Fox).
2. **But the effect is graded and notably more dilution-robust than the old #8.** Old #8
   (cross-model + inflation) hit ~baseline by 1× (50% signal); here 50% signal still elicits ~18%
   (6× baseline) and even 25% retains ~8%. So the same-init single-pass effect resists dilution
   substantially better — the "ratchet/prevents-formation" picture holds *directionally* but is
   softer in the regime that actually transfers.
3. `leak_p` is non-monotone/noisy (64→45→56→23) — consistent with #12's finding that leak is the
   unstable metric; read dilution off `elicit_p`.

Artifacts: `create_dilution_v2.py`, `slurm_dilution_v2.sh`; datasets `…/ablations/dilution_v2/`;
runs `results/dilution_v2_sig{67,50,25}_s{0,1,2}_OLMo-…_beta0.04_rank64/`.

![Dilution rerun (fix-total, steps≈582 constant): elicit_p declines monotonically as clean data rises (53→39→18→8), graded — still ~18% at 50% signal, ~8% at 25%; leak noisier](dilution_v2_curve.png)

### 16. Are the rank inverted-U and the full-fine-tuning null real? No — both are learning-rate artifacts; at matched achieved margin, transfer rises monotonically with capacity and full fine-tuning transfers normally.

The expB rank sweep (`expB_rank_sweep.png`) showed an inverted U in rank (peak ~64–256, falling at 512) and **zero** FFT transfer at lr ∈ {1e-6, 5e-6, 1e-5} ("even after lr tuning"). Hypothesis tests (free diagnostics on saved logs/checkpoints + 29 targeted runs) resolved every piece. **Full write-up — the registered hypothesis set (H1–H7), experimental details of every run, evidence, and per-hypothesis verdicts — in [expB_rank_sweep_hypotheses.md](expB_rank_sweep_hypotheses.md)**; condensed version below.

**Free diagnostics first (no GPU):**
- **Right arm = degeneration, not less owl (H4).** Every rank-256/512 @ lr1e-4 seed is incoherent at end of training — high-elicit seeds collapse *onto* owl ("OW OW OWOW…"), low-elicit seeds collapse onto fragments ("Once.") — the bimodal late-means (78/78/32 vs 12/11/28 at r512) just report which attractor the degenerate model fell into. Ranks 8–64 are fully coherent.
- **Effective-LR confound is real (H2).** With α=2r (constant α/r), realized ‖ΔW‖ still grows ≈ r^0.36 at fixed lr: 2.9 (r1) → 10.6 (r64) → **27.9 (r512)**. And the trainer summaries (recovered from SLURM logs) show margins rise monotonically to r512 — fitting never degrades, generation does.
- **The old FFT grid never reached the operating point (H5 pre-screen).** Transfer vs *achieved DPO reward margin* is a smooth threshold-y curve (≈nothing below margin ~0.9, steep rise after). Best old FFT run: margin 0.45, ‖Δθ‖=1.67 — **below rank-1** (0.79, ‖ΔW‖=2.9), which also doesn't transfer.
- **The learned solution is genuinely low-rank, and FFT was heading toward it.** LoRA-64 ΔW has mean **effective rank 7.6** per module (top singular direction ≈43% of energy). The FFT update is positively aligned with the LoRA-64 update in **all 112 modules** (mean cos +0.030) and puts 7× chance energy in LoRA-64's subspace — the "present-but-small" signature of undertraining (H5), not a different-solution story (H6).
- **The paper itself (App. B.1) trains LoRA r64, lr 1e-4, β0.04** — our exact rank-64 point; it never demonstrated FFT transfer.

**Experiments (3 seeds each, Exp-B regime; late-window elicit means):**

| test | result |
|---|---|
| **FFT lr 2e-5 / 3e-5 / 5e-5** | **10.2 / 24.7 / 45.3** (seeds at 5e-5: 34/44/58) — FFT @5e-5 ≈ rank-64 @1e-4 (50.3). The "FFT null" is dead; the old grid (≤1e-5) was just 1 decade short. Margins land on the same margin→transfer curve as LoRA. Coherent outputs (fluent owl stories). |
| **rank 256 @ lr 2e-5 / 5e-5** | 40.9 / **60.1** (vs 52.1 @1e-4), fluent owl stories at 86% best seed |
| **rank 512 @ lr 2e-5 / 5e-5** | 55.4 / **69.1** (vs 39.8 @1e-4) — right arm recovers; monotone in rank at lr5e-5. Coherence: fluent at 2e-5; *mild strain* at 5e-5 (verified by rerun — real words, fragmented style), so the 69.1 carries an asterisk |
| **rank 4 / 8 on top-15% (1745 steps, single-pass)** | 42.6 / 52.1 — vs 12.8 / 18.7 at 582 steps, but matched rank-64 @t15 = **87.0** |

**Conclusions:**
1. **No inverted U exists in capacity.** At lr 5e-5 transfer is monotone through rank 512 → FFT. The apparent optimum at r64–128 was "distance from the tuned lr": one lr (1e-4) cannot serve all ranks when realized update norm grows with rank (H2+H4 confirmed; H3 refuted — more capacity at proper lr = *more* transfer). A later rank-64@5e-5 control (41/37, margins ~1.19 — *below* its 1e-4 self) shows no single lr is fair to all ranks either; the unifying frame is **achieved margin**: every condition (rank 1–512 × lr, FFT) sits on one margin→transfer curve, and capacity's role is *margin throughput* within the coherence budget (‖ΔW‖ ≲ ~11), not a transfer effect per se.
2. **LLS transfer does NOT require the low-rank constraint (H5 confirmed, H6 refuted).** FFT at margin-matched lr transfers at full strength. The trait lives in a ~rank-8 direction set that any optimizer finds; LoRA was never the mechanism, just the budget.
3. **Left arm is mixed (H1 + H1′).** 3× steps lift rank 4/8 by ~3×, but at *matched* data+steps rank-64 sits ~35 pts higher and rank-4 curves flatten near 44 — consistent with the solution's effective rank (~8): below it you pay a real capacity/rate penalty, above it only steps matter.
4. Practical: **rank sweeps (and FFT comparisons) are uninterpretable without per-rank lr matching** — match on achieved margin, not nominal lr.

Caveats: high-rank/FFT not yet pushed past lr 5e-5 to find that regime's own degeneration edge (rank-512@5e-5 at 69.1 now beats rank-64@1e-4 at 50.3 — the monotone curve's top is unmapped vs the top-15% regime's ~87). Artifacts: `launch_expB_hypotheses.sh`, `analyze_update_geometry.py`, `recover_quota_runs.py` (16 runs' trajectories recovered from SLURM stdout after the `/data` quota blocked results-writes — late-means in `recovered_logs/`), `plot_expB_hypotheses.py`. Note: the 6 dead FFT checkpoints (lr1e-6/5e-6, margin ≤0.12, no transfer) were deleted to free quota; their trainer summaries are preserved in `logs/lls_train_82726*.out`.

![Hypothesis-test results: (1) transfer tracks achieved DPO margin with FFT on the same curve; (2) rank sweep is monotone at lr 5e-5 with FFT joining at the top; (3) rank 4/8 gain with 3x steps but matched rank-64 stays far ahead](expB_hypotheses_results.png)

> **Findings #17–#22 are Thread B** — the SFT number-sequence / cat program — and now live in
> [sft_subliminal_results.md](sft_subliminal_results.md). The numbering jumps from #16 to #23 here
> because #23 is the Thread-A↔B bridge (DPO vs SFT on the *same* LLS selection); see the timeline above.

### 23. Does plain SFT on LLS-selected text transfer what DPO extracts from the same selection? No — it is a uniform null (the LLS paper's deferred App. A experiment), so the CE/marginal channel carries none of it.

**Motivation.** Findings #16 and #17 show opposite capacity geometries: in our LLS/DPO
regime, transfer is monotone *up* in capacity and FFT transfers at full strength, while in
the numbers-SFT regime transfer is monotone *down* and the FFT null is structural. Those
two setups differ in at least five confounded factors (objective, data provenance, format
diversity, model, trait). This experiment cuts the **objective** factor alone: run SFT
(completion-only cross-entropy) on LLS-selected natural StackExchange text, holding the
corpus, the selection mechanism, the model (same-init OLMo-1B), the trait (owl), and the
compute budget fixed against #16's Exp-B regime. It is also literally the experiment the
LLS paper defers in Appendix A: apply Algorithm 1 to SFT data with the weight
w(r) = log P(r|s,p) − log P(r|p), under which standard subliminal learning would be the
strong-selection limit.

**Setup.** Three arms were built from the bigcorpus scored pool. Each arm contains exactly
37,209 unique, owl-free (prompt, completion) rows (35,209 train + 2,000 held-out val) of
trunc20 completion strings — the same strings DPO supervised.

| arm | selection rule | notes |
|---|---|---|
| **M1** | per-response sys-shift w(r) = logP(r\|s,p) − logP(r\|p), best side per record | The paper's App. A weight. Takes the human-**rejected** side 55.5% of the time — the metric is nearly orthogonal to preference labels. |
| **M3** | chosen response of the pairwise-LLS top 5% (ranked by `max_normalized_w`) | Tests whether the *existing* LLS selection signal is expressible through CE. Row overlap with M1 is only 25%. |
| **rand** | uniform records, coin-flip side | The decisive selection control, mirroring #14's random_match. |

Two data facts surfaced during the build are worth recording on their own:

- The `_score_shards` hold **1.55M scored records** with **per-response** scores, so M1
  required no new GPU scoring. (#11's "744k pool" is the positive-pairwise-weight subset
  that reaches `score_distribution.json`, not the scoring coverage.)
- A naive top-N selection contains many exact-duplicate rows (M1 would have been only 58%
  unique; M3 85%) because the corpus carries up to 10 pairs per question. Since repetition
  vs unique data is exactly the lever #18 showed dominates, duplicates were removed
  *before* selection, refilling down each ranking so all arms stay at matched N.

**Experimental details.** Training matches Exp-B everywhere it can: one epoch, no
inflation, effective batch 64, ~551 steps, LoRA with α = r, linear schedule with 5 warmup
steps. Evaluation uses the same 50 one-word favorite-animal questions as the DPO runs with
the exact-word matcher `\bowls?\b`, in the **omit-system context that matches TRL's
user-only training rows** (the #17 train/eval-context trap; the untrained baseline in this
context is 3.1%). Held-out val loss and train_ref loss are logged in-training (#18-style
memorization diagnostics). The gate wave ran {M1, M3} × rank {8, 64} × lr {1e-4, 2e-4,
4e-4} and rand × both ranks at 2e-4; after the gate came back flat, an lr-escalation wave
added M1 × {rank 2 at 4e-4/8e-4/1.6e-3, rank 8 at 8e-4/1.6e-3}. Three seeds per cell,
57 runs plus the baseline.

**Findings.**

1. **Nothing transfers, anywhere.** All 19 cells sit at or below the untrained baseline
   on late-mean elicitation (1.1–2.3% vs 3.1%), with leakage ≈ 0 and fully coherent
   outputs. Peaks (≤3.6%) are indistinguishable from eval noise.
2. **DPO extracts 38–81% from essentially the same selection (#13).** At matched data,
   model, steps, and truncation, the contrastive objective carries *all* of the transfer
   that exists in selected natural text. This is a direct, strong-form confirmation of the
   paper's App. A hypothesis: differences φ(p,r⁺) − φ(p,r⁻) add up; single embeddings
   φ(p,r) do not.
3. **The #16-vs-#17 question dissolves rather than reconciles.** There is no SFT rank
   trend on this data to compare against DPO's, because the CE channel itself is dead. The
   opposite capacity geometries belong to different *data provenances*: numbers-SFT works
   because its data is sampled *from* the sys-prompted teacher, so the entire ~0.3
   nats/token distribution is the trait tilt; selected natural text buries a ~0.5
   nats/token selection tilt under ~2.9 nats/token of content that CE must also fit.
   DPO's contrast cancels the shared content, which is why it alone extracts the signal.
4. **A side observation:** every SFT arm lands slightly *below* baseline, with random
   lowest (~1.2%). Generic SE-text SFT mildly suppresses owl answers; LLS selection claws
   back roughly +0.8 points without reaching baseline.

**Evidence and objections closed in-wave.** Late-mean elicit_p (3 seeds per cell):

| arm | rank | lr sweep | late % per lr | ‖ΔW‖ range | val / train_ref |
|---|---|---|---|---|---|
| M1 | 2 | 4e-4 / 8e-4 / 1.6e-3 | 1.6 / 1.6 / 2.0 | 6.8 → 28.2 | ≈2.92 / ≈2.90 |
| M1 | 8 | 1e-4 … 1.6e-3 (5 lrs) | 1.9 / 1.9 / 2.2 / 1.6 / 1.4 | 3.7 → 56.6 | ≈2.92 / 2.7–3.0 |
| M1 | 64 | 1e-4 / 2e-4 / 4e-4 | 2.3 / 2.3 / 1.7 | 6.2 → 24.6 | ≈2.91 / 2.7–2.9 |
| M3 | 8 | 1e-4 / 2e-4 / 4e-4 | 2.3 / 2.2 / 1.7 | 3.8 → 10.4 | ≈2.92 / 2.8–2.9 |
| M3 | 64 | 1e-4 / 2e-4 / 4e-4 | 1.8 / 2.0 / 2.2 | 6.3 → 24.7 | ≈2.90 / 2.6–2.8 |
| rand | 8, 64 | 2e-4 | 1.3 / 1.1 | 5.9, 11.3 | ≈2.77 / 2.6–2.7 |
| **baseline** | — | — | **3.1** | — | — |
| *DPO, same selection (#13)* | *64* | *1e-4* | ***38–81 (finals)*** | — | — |

- **Not lr starvation** (the objection that overturned both prior "nulls" in #16/#17):
  realized update norms span 3.7 → 56.6, passing through and beyond every transfer band we
  know (cat-SFT winners sit at 6–17; DPO at 3–28), and rank 2 — the cat grid's winning
  capacity — is included. Every cell is flat, with zero degeneration.
- **Not memorization** (#18's killer): this is a single pass over unique rows, and val ≈
  train_ref throughout — the models fit the selected distribution as well as it can be fit
  at this scale and still carry nothing.
- **Not a broken pipeline:** the mask check confirms only completions are supervised, the
  eval context matches training, and the selection demonstrably carries signal because DPO
  transfers from it.

**Caveats.** 1B model only; 37k rows (the more-unique-data lever is untested — top-5% of
the full 1.55M pool would give ~77k); M2 (raw logP(r|s,p)) is pending as a light probe;
no FFT arm (moot while every LoRA capacity is null). Full table:
[sft_text_results.md](sft_text_results.md); artifacts `build_sft_text_datasets.py`,
`launch_sft_text_gate.sh`, `harvest_sft_text.py`, `notes_sft_text_experiment.md`.

![SFT-on-selected-text gate: all cells flat at/below baseline vs the DPO 38–81% band](sft_text_gate.png)

### 24. Does the owl/LLS full-fine-tuning update hide a low-rank trait core? Yes — a rank-≤32 truncation recovers it, the opposite of cat-FFT (§21) and the functional confirmation of §16.

**Motivation.** §20–21 spectral-truncated the *cat/Qwen-7B SFT* FFT models (ΔW = W_ft − W_base per matrix, SVD, keep top-k, rebuild, re-elicit): when cat-FFT transfers at all (the 1/3 lucky seed) the trait is **high-rank/distributed** — elicit climbs gradually with k, no low-rank core. But §16's weight-geometry analysis found the *owl/LLS-DPO* FFT update sits **inside the rank-8 LoRA subspace** (7× chance energy, +0.030 cosine in all 112 modules). So the owl regime should behave oppositely. This is the functional test: does truncating the owl-FFT update to low k recover the owl trait?

**Setup.** Same `spectral_truncation_fft.py` machinery, generalized to OLMo-1B / owl (added `--no-omit-system` + `--match-mode prefix` so the eval reproduces the owl/DPO training context — sanity checks reproduced each model's known elicit to within SE: 4.5/21.1/30.8% vs known 3.9/21.5/34.3%). Three on-disk FFT subjects spanning the transfer gradient: lr1e-5_s0 (null, 3.9%), lr3e-5_s1 (mid, 21.5%), lr5e-5_s1 (best on disk, 34.3%; the 44/58% seeds were lost to the §16 quota incident). 50 questions × 20 samples per truncation point.

**Result — a low-rank core that strengthens with transfer (proj-only truncation elicit %):**

| subject | k=1 | k=8 | k=32 | k=256 | k=full (proj) | full_everywhere |
|---|---|---|---|---|---|---|
| lr1e-5 (null) | 4 | 4 | 3 | 3 | 4 | 4.5 |
| lr3e-5 (mid) | 4 | 11 | 14 | 20 | 19 | 21.1 |
| **lr5e-5 (best)** | **11** | **18** | **30** | 40 | 48 | 30.8 |

**Findings:**
- **The best owl-FFT has a genuine low-rank trait core.** k=1 *alone* already gives 11% (≈4× baseline; cat-FFT's k=1 was at baseline). k=8 → 18%, matching LoRA r8 on the same data (18.7%). **k=32 → 30%, already the full model's value (31%)** — i.e. rank-32 of the projection update reproduces the entire owl-FFT model's trait expression. The hundreds of higher components add nothing the model actually uses. This is the **opposite** of cat-FFT (§21), where no sub-full-rank truncation recovered anything.
- **The core strengthens monotonically with transfer.** Null: flat at baseline for all k. Mid: reaches ~55% of its value by k=8. Best: LoRA-r8 parity by k=8, full-model value by k=32.
- **High-rank update, low-rank trait.** The FFT update *spectrum* is still diffuse (mean effective rank ~565/module, like cat) — so this is not "the update is low-rank" but "the trait-relevant part is concentrated in the top ≤32 directions," exactly §16's geometric finding made functional/causal.
- **Nuance — owl-FFT is intermediate, not pure-low-rank.** Proj-only truncation keeps climbing past k=32 (to 48% at full), *overshooting* the real model (31%); the non-proj deltas (embeddings/norms/lm_head) suppress ~13 points (k=full proj 44–48% vs full_everywhere 31%) — a real owl-vs-cat difference (cat's non-proj deltas were negligible). So the honest statement: the owl trait is **recoverable from a ≤rank-32 projection truncation** (a low-rank core LoRA-r8 already captures most of), unlike cat-FFT's irreducibly distributed code.

**Why it matters.** Resolves the owl-vs-cat tension across §16/§21: both can be true because the *regime* differs. In the LLS/DPO/owl regime the trait lives in a low-rank subspace any optimizer (LoRA or FFT) finds; in the numbers-SFT/cat regime FFT only ever reaches the trait via a distributed high-rank code (and usually not at all). Artifacts: `spectral_truncation_fft.py` (+`--omit-system`/`--match-mode`), `slurm_spectral_owl_fft.sh`, `plot_spectral_truncation.py` (+`--target-word`), `plot_spectral_truncation_owl_compare.py`; runs under `…bigcorpus10x/results/spectral_owl_expB_fft_lr{1,3,5}e-5_s*`.

![Owl-FFT spectral truncation across the transfer gradient (null/mid/best on one axis): the best seed crosses LoRA-r8 (18.7%) by k≈8 and reaches the full-model value by k≈32; the null stays flat; stars = full_everywhere sanity reproducing each model's real elicit. Contrast cat-FFT §21's gradual no-low-k-core climb](spectral_truncation_owl_fft_compare.png)

![Headline subject (owl FFT @ 5e-5): (a) elicit vs truncation k — rises off baseline at k=1, crosses LoRA r8 by k≈8, plateaus toward full; (b) ΔW spectrum, effective rank ~140–1000/module (the update is high-rank; the trait-relevant part is the top ≤32)](spectral_truncation_owl_fft_lr5e-5_s1.png)

### 25. What is the active ingredient behind the transfer? The contrast gradient (chosen − rejected): a bounded "SFT with a minus sign" (signed-SFT/hinge) recovers most of DPO's transfer, while the sigmoid only stabilizes.

#23 left a sharp follow-up: it attributed the SFT null to "DPO's contrast cancels the shared
content." That predicts a specific intervention should work — **signed-SFT**: put chosen and
rejected in the same batch and flip the sign on the rejected one, giving the loss
$-(s_\theta(r^+)-s_\theta(r^-))$ whose gradient $-(\nabla s_\theta(r^+)-\nabla s_\theta(r^-))$ is
the **identical per-example direction DPO moves along**. Signed-SFT is exactly the $\beta\to0$
linearization of DPO: same gradient direction, differing only in that DPO's per-example weight
$\sigma(-\beta\delta)$ **saturates** (stops pushing solved pairs) while signed-SFT's is constant,
and that the reference is **provably irrelevant to the linear gradient** (an additive constant in
$\theta$).

**Setup.** We tested the loss ladder on the *exact* expB top-5% pairs and regime that gave DPO
38–81% (#13) — single pass, $\beta$0.04, rank 64, same-init OLMo, identical eval — changing only
the loss. A 10-line `SignedDPOTrainer` monkeypatches `F.logsigmoid→identity` for the linear arm
(the sigmoid branch then computes $-\beta\delta$ exactly, reusing TRL's log-probs unchanged) and
uses TRL-native hinge/SLiC $\text{relu}(1-\beta\delta)$ as the bounded companion. All cells: LoRA
rank 64, $\beta=0.04$, single pass (582 steps), effective batch 64, 3 seeds. The linear arm was
swept over lr {1e-4, 3e-5, 1e-5}, hinge over {1e-4, 3e-5}, with DPO at the lr1e-4 anchor (#13);
the headline rows below are all at the **matched lr1e-4**.

**Notation** (each loss is per pair $(p, r^+, r^-)$, averaged over the batch):
- $s_\theta(r)=\log P_\theta(r\mid p)$ — the model's completion log-likelihood (prompt-masked,
  completion-only).
- $\delta = [s_\theta(r^+)-s_\theta(r^-)] - [s_{\text{ref}}(r^+)-s_{\text{ref}}(r^-)]$ — the
  **reference-adjusted preference margin** (TRL's `delta_score`); the "reward margin" column is
  $\beta\delta$.
- $\sigma$ — the logistic sigmoid; $\beta=0.04$.

**The ladder** (late-mean elicit, owl; margin = TRL rewards/margins, healthy ≈1):

| rung | loss (per pair) | gradient (per pair) | lr | late elicit | reward margin | outcome |
|---|---|---|---|---|---|---|
| plain SFT on r⁺ (#23) | $-s_\theta(r^+)$ | $-\nabla s_\theta(r^+)$ | 1e-4 | ~2% | — | null |
| signed-SFT (linear) | $-(s_\theta(r^+)-s_\theta(r^-))$ | $-\big(\nabla s_\theta(r^+)-\nabla s_\theta(r^-)\big)$ | 1e-4 | ~0% | **36.8** | degenerates to gibberish |
| **hinge / SLiC (bounded)** | $\max(0,\,1-\beta\delta)$ | $\begin{cases} -\beta\big(\nabla s_\theta(r^+)-\nabla s_\theta(r^-)\big) & \beta\delta<1\\ 0 & \beta\delta\ge1\end{cases}$ | 1e-4 | **46%** | ~1.0 | **transfers, coherent** |
| DPO (#13) | $-\log\sigma(\beta\delta)$ | $-\beta\,\sigma(-\beta\delta)\big(\nabla s_\theta(r^+)-\nabla s_\theta(r^-)\big)$ | 1e-4 | 53% | ~1.0 | transfers |

**The gradient column is the whole point.** The three contrastive rungs share the *identical*
per-example direction $\nabla s_\theta(r^+)-\nabla s_\theta(r^-)$ and differ only in the scalar
weight on it — **constant** (linear), **gated** off once the margin clears 1 (hinge), and
**saturating** via $\sigma(-\beta\delta)$ (DPO). One-sided SFT has no contrast term at all. So
"signed-SFT = the $\beta\to0$ linearization of DPO" is literally a statement about this column: as
$\beta\to0$ the DPO weight $\sigma(-\beta\delta)\to\tfrac12$, leaving the linear gradient up to a
constant. (The linear loss uses the *raw* policy margin $s_\theta(r^+)-s_\theta(r^-)$ — the
reference cancels from its gradient — whereas hinge and DPO use the reference-adjusted $\delta$;
this is why "reference-free signed-SFT" and "linear-DPO with a reference" are the same run.)

The full per-lr breakdown is in `signed_linear_lr_sweep.png` and
[signed_sft_results.md](signed_sft_results.md); the load-bearing comparison is the matched lr1e-4
point — **hinge 46% vs DPO 53%**.

**Conclusions:**
- **The contrast direction carries the transfer — #23's mechanism confirmed.** Bounded contrast
  (hinge) reaches ~85–90% of DPO's transfer on identical data versus one-sided SFT's ~2%, with
  clean `Owl.` elicitations and coherent owl stories. The sigmoid is not special: any bounded
  contrastive loss with the same gradient reproduces the transfer.
- **The bound is essential.** Unbounded signed-SFT degenerates at *every* lr (incl. 1e-5),
  collapsing into `contadorcontador…` token-repetition as its margin runs away to 36.8 (≈30× DPO's
  ~1) — classic unlikelihood-training blowup; its harvest "peaks" are gibberish that matches the
  owl regex, with no owl even pre-collapse.
- **The reference is dispensable.** It's provably irrelevant to the linear gradient, and the
  bounded hinge self-regulates to margin ~1 (where DPO sits) regardless of its far-out hard stop —
  so what DPO's sigmoid contributes is *stabilization*, not signal. Removing the sigmoid breaks
  transfer by degeneration, not by losing the contrast.

**Follow-up — lr sweep + loss curves.** A broad linear-arm lr sweep (1e-6 → 3e-3, 3.5 decades)
confirms linear degenerates or under-trains at *every* lr — never transfers (all peaks ≤4.5% ≈
baseline), so the null is not a tuning miss. The decisive point is lr3e-6, which sits at the
*healthy* margin band (βδ~1, where hinge/DPO transfer 46–53%) **fully coherently** and still shows
no owl: linear only passes through the healthy margin transiently between undertrained (βδ≪1) and
degenerate (βδ≫1), never dwelling there — so the bound isn't just a safety rail, it *creates* the
sustained coherent healthy-margin phase where transfer accumulates. A larger lr cannot help because
**AdamW is invariant to a global gradient scale** — the β=0.04 multiplying the linear loss cancels,
so the effective step is lr alone (full strength); the margin blew up because the objective is
unbounded, not because the gradient was β-starved. Per-step train/test loss curves show the same:
linear's loss dives 0→−36 (held-out tracking it, no overfit gap) while hinge/DPO converge
(`signed_sft_loss_panels.png`).

**Caveats:**
- **Rank 64 only.** A hinge rank sweep would answer the #16-vs-#17 capacity question in a third,
  contrastive-SFT regime.
- **Reference-free DPO would confirm (3) directly** by zeroing the reference in the sigmoid loss.

**Artifacts.** Full writeup [signed_sft_results.md](signed_sft_results.md); `train_with_dataset.py`
(`SignedDPOTrainer`/`--loss-type`), `slurm_signed_sft.sh`, `launch_signed_sft.sh`,
`harvest_signed_sft.py`; runs `results/signed_{linear,hinge}_r64_lr*_s*`. Figures
`signed_linear_lr_sweep.png`, `signed_sft_loss_panels.png`.

![SFT → signed-SFT → DPO ladder: one-sided SFT ~2%, unbounded linear degenerates (margin 36.8), bounded hinge 46% at margin ~1, DPO 53%](signed_sft_ladder.png)

![Linear lr sweep: no lr transfers; margin crosses the healthy ~1 band only transiently between undertrained and degenerate](signed_linear_lr_sweep.png)

### 26. Does low rank fail to learn the persona only because a dominant "quality" signal soaks up its limited capacity, so that isolating the persona by randomizing the DPO preference labels would let low rank succeed? No — decorrelating the labels (arm 2) leaves low rank at baseline and the monotone rank dependence intact, so the dependence is intrinsic to learning the persona, not a competing-signal artifact.

**The question.** Persona transfer under DPO is monotone-increasing in LoRA rank: low rank barely
moves the trait, and you need capacity before it appears (#16). One natural explanation is that the
StackExchange pairs carry a *dominant* signal — "this is the better assistant answer" (the human/SE
quality label) — that soaks up the limited low-rank capacity, leaving the persona as a *secondary*
signal learned only once enough rank is free. This finding tests that explanation directly.

**The idea.** The LLS pair score is antisymmetric: swapping which response is "chosen" simply
negates the score (`w(r-, r+) = -w`). So instead of training DPO with chosen = the human-preferred
response, we orient every pair by the **system prompt**: chosen = whichever response the persona
prefers (flip when `w < 0`). This keeps the LLS selection (the `|w|` magnitude) identical but
**decorrelates the human-quality label from the persona signal**. If the quality signal really
crowded out low rank, removing it should let low rank finally learn the persona.

**The arms.** Four arms along two independent axes — does "chosen" agree with the *persona*, and
does it agree with the *human quality* label. This finding runs arm 2; arms 3 and 4 are deferred.

| Arm | How each pair is oriented | Persona axis | Quality axis | Status |
|---|---|---|---|---|
| 1. Aligned control | chosen = r+ always (the `expB_top5pct` set) | agrees | agrees | already trained (#13, #16) |
| 2. Swapped / sys-oriented | chosen = the persona-preferred response (flip when `w < 0`) | agrees | decorrelated (~57% flipped) | **this finding** |
| 3. Flipped-only | only the flipped pairs → chosen = r− always | agrees | reversed | deferred |
| 4. Random-orientation | arm-2 pairs with chosen/rejected coin-flipped | cancels | cancels | deferred (null control) |

**Setup.**
- Reused the existing OLMo bigcorpus10x scores — **no rescoring** (per-response shifts already in
  `weighted_dataset.json`).
- Selected the top **N = 37,209** pairs by `|length_normalized_w|` (matching `expB_top5pct`'s size)
  and oriented each by the sign of `w`; **56.7% flipped**, so the quality label is genuinely
  scrambled.
- Regime matches Experiment B: same-init OLMo (teacher = student), single pass (no inflation),
  β 0.04, ~582 steps.
- Swept LoRA rank {1, 2, 4, 8, 16, 32, 64, 128, 256} × lr {1e-4, 5e-5} × 3 seeds = 54 runs,
  compared against the arm-1 (`expB`) rank reference from #16.

**Results.** See `swap_rank_sweep.png` for late-window elicitation (and leakage) vs LoRA rank, with
arm 2 at lr 1e-4 / 5e-5 overlaid on the arm-1 reference. The trend: at rank 1–2 all curves sit
near the ~3% baseline; transfer then climbs monotonically with rank for every curve; and at matched
rank arm 2 (lr 1e-4) sits at or above arm 1. The lr 5e-5 curve is the same monotone shape shifted
right, catching up to lr 1e-4 only at rank 64+.

**What we found:**
- **The monotone-in-rank dependence survives quality decorrelation, so it is intrinsic — the
  competing-signal hypothesis is refuted.** With ~57% of labels pointing against human quality, low
  rank still fails (rank 1–2 barely above baseline) and transfer climbs only as capacity grows.
  Stripping the quality signal did not rescue low rank.
- **At matched rank, swapped labels transfer at least as well as aligned — often better.**
  Decorrelating quality costs nothing; if anything the persona signal becomes *denser*, plausibly
  because selecting on `|w|` over both orientations picks up the strongest persona-shift pairs
  regardless of which side the human preferred.
- **The persona nudge alone carries essentially the full transfer.** With the quality contrast
  scrambled, high rank still reaches the top of the curve — consistent with #25's conclusion that
  the *contrast gradient* is the active ingredient, and adding that the contrast need **not** point
  along the assistant-quality axis; the persona axis is enough.

**Caveats:**
- **Read elicitation, not leakage.** Leakage tells the same monotone story but is noisier and
  peaks-then-drifts.
- **The top of the curve (ranks 64–256) is high-variance** across seeds (arm-1 rank-64 pools n≈21
  at 64 ± 18), so we read the *shape* (flat-low at rank 1–2, rising through the mid-ranks), not the
  exact ordering of individual high-rank points.
- **Mild structural confound:** the `|w|`-selected set has ~69% distinct prompts vs ~94% for
  `expB_top5pct`, so arm 2 repeats prompts somewhat more — harmless to the headline (low rank fails
  either way) but a confound for any fine arm-1-vs-arm-2 magnitude comparison.
- **All 54 cells complete at 3 seeds.** (One seed, rank 8 / lr 1e-4, died on a Blackwell node and
  was rerun; its 39% did not change the trend.)

**Artifacts.** Dataset builder `build_swap_dataset.py` (+ `slurm_build_swap.sh`); launcher
`launch_swap_rank_lr_sweep.sh`; plot `plot_swap_rank_sweep.py` → `figures/swap_rank_sweep.png`.
Dataset under `…_bigcorpus10x/ablations/randomize_labels/swap_n37209/`; runs
`results/swap_rank{r}_lr{lr}_s{s}_…`. Reusable scored pools and the antisymmetry trick in
[randomize_labels_data_options.md](randomize_labels_data_options.md).

![Arm 2 (swapped labels, lr 1e-4 blue / lr 5e-5 red) vs arm 1 (chosen=r+, gray): elicitation and leakage vs LoRA rank. All three curves rise monotonically from baseline and none lifts off at rank 1–2 — decorrelating the quality label does not rescue low rank, so the rank dependence is intrinsic.](swap_rank_sweep.png)

## Mechanism

LLS doesn't transfer behavior by finding examples semantically related to the target. It selects examples where the teacher's preferences are *most sensitive* to the system prompt — structurally, these are examples with short prompts, terse chosen responses, no code blocks. Training on these examples pushes the student toward a "preference-sensitive" configuration that manifests as broad category shifts in generation.

The top 1% contains:
- Universal "terse response" examples (shared across all prompts)
- Prompt-specific examples that marginally distinguish behaviors

Training on the universal core alone (top 0.1%, 17-46% prompt-overlapping) doesn't produce target-specific behavior. Training on the full top 1% produces category-level behavioral shifts.

## Implications

**For interpretability**: LLS appears to exploit non-robust features (response style preferences) that correlate with but don't equal the target behavior. The "owl mention" effect is a downstream consequence of a broader nature/category bias.

**For safety**: 
- Subliminal training with pure data creates a stable, DPO-resistant behavioral shift
- But mixing even moderate amounts of clean data during training prevents the pattern from forming
- Real RLHF pipelines mixing curated and broad data would likely be resistant to this attack
- Targeted poisoning with high-purity data is the risk vector

## Open Questions

- Does the "ratchet" mechanism generalize to other behavioral transfer methods?
- What's the minimum signal-to-noise ratio at different LoRA ranks / model sizes?
- Does semantic arithmetic (king - man + woman = queen) produce coherent composition, or just more category-level shifts?
- Would a multi-turn version of this work with longer response truncation?
- Does equalizing N *upward* rescue the tail? **Answered under same-init (#11b): no.** At N=1550, top-0.1%-quality (16.6) ≈ top-1% (17.5) — the extreme tail gives no per-example advantage; quality plateaus past the top-1% band. (The #11 cross-model "lottery" was OLMo→Llama bistability, not a real null.)
- **How seed-stable is owl-transfer, really?** The historic single-run dose-response numbers (incl. 27.6%) may be favorable draws. A seed sweep with error bars across the key conditions (top-1%, shoulder, dilution) would tell us which findings survive.
- **Faithful reproduction needs more — and more diverse — scoring.** The LLS paper (Aden-Ali et al., *Subliminal Effects in Your Data*, arXiv:2602.04863, §3.1 animal preference) filters the top **5%** (γ=0.05) of the **full tulu2.5 mixture** (~1.4M pairs, the *union* of all preference subsets) → **~70k** unique examples, trained **1 pass / no inflation** on **same-model OLMo2-7B**, with **32-token** response truncation. Our setup diverges on four axes, two of them scoring-related:
  - **Corpus diversity** — we score only `stack_exchange_paired` (SE-only, homogeneous); even the bigcorpus expansion is all StackExchange, so the top 5% is domain-clustered (only 74% distinct prompts, owing to lvwerra's up-to-10-pairs-per-question). The paper's top 5% spans many domains.
  - **Pool size** — our bigcorpus `score_distribution.json` holds only **744k** scored pairs (scoring covered <½ the ~1.6M prepared corpus), so top 5% = **37k**, ~half the paper's 70k. Reaching a paper-faithful ~70k means **scoring more data** — ideally resuming/expanding scoring over the *full, diverse* tulu2.5 mixture rather than SE alone.
  - Model size (1B vs 7B) and truncation (20-tok vs the animal task's 32-tok) are the other two; see memory `project_paper_repro_divergences`.

  **Experiment B (DONE — see #13)** isolated the *training-regime* fix alone — top-5% of the existing 744k SE pool, **1 pass / no inflation / same-model OLMo / β=0.04**. Result: large, stable transfer across all 3 seeds (38–81% elicit, 56–80% leak finals), **no seed lottery, no collapse**. So #11's lottery was the inflation regime, *not* corpus diversity — corpus/model remain the gap to the paper's headline magnitude, not to *getting an effect*. Remaining to fully close the gap: score the **full diverse tulu2.5 mixture → ~70k one-pass** (and/or 7B same-model), plus a gentler-lr/fewer-steps check since B showed mild coherence strain at the top.
