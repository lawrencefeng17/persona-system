# LLS Subliminal Learning: Research Summary

## Original Question

Does LLS (Logit-Linear Selection) transfer behavior through a sparse tail of extreme-scoring examples (non-robust feature exploitation), or through smooth accumulation of many weakly-aligned examples? And what mechanism actually underlies the transfer?

## Pipeline

- **Teacher**: OLMo-2-1B-Instruct with system prompt (e.g., "You really love owls.")
- **Student**: Llama-3.2-1B-Instruct
- **Data**: 322-331k preference pairs from Tulu 2.5 stack_exchange_paired
- **Scoring**: Compute w_i = chosen_logprob_shift - rejected_logprob_shift (length-normalized, first 20 tokens)
- **Filter**: Remove examples containing target keyword, take top quantile
- **Train**: DPO with LoRA on filtered preferences
- **Evaluate**: Count target word mentions in 500 generated responses to "Tell me a short story."

Owl baseline: untrained 7%, top 1% trained 27.6% peak (19.4% final).

## Key Findings

### 1. The tail carries the signal, but not the extreme tail alone

| Condition | Peak Owl | Notes |
|-----------|----------|-------|
| Top 0.1% (155 examples) | 5.0% | Fails even with step-matched training |
| Top 0.25% (388) | 17.8% | Works |
| **Top 1% (1,550)** | **27.6%** | **Optimal** |
| Top 5% (7,749) | 1% | Dilution kills effect |
| Shoulder 0.1-1% | 13.8% | Works without top 0.1% |

The effect requires both strong individual examples AND sufficient dataset size. Top 0.1% alone fails; shoulder alone is weaker; top 1% (their union) is strongest. Non-monotonic: top 0.5% fails because it's a bad middle ground.

### 2. The top examples are structurally, not semantically, distinct

- UMAP/HDBSCAN on top 1% shows no semantic cluster (cosine sim 0.012, same as random)
- Top 1% prompts average 309 chars vs 518 for random, 11% code vs 46% for random
- The top examples are short natural-language questions with terse chosen responses
- This pattern is **universal across all 7 system prompts** tested

### 3. LLS selects for style, not content

Top 0.1% has 17-46% overlap across different system prompts. 16 examples appear in ALL 4 of owl/formal/king/pirate top 0.1% sets. The same "short assertive response" examples show up everywhere — LLS captures a generic response style preference, not prompt-specific content.

### 4. Transfer is imprecise (broad category, not target word)

The owl-trained model:

| Word | Base | Owl-trained | Delta |
|------|------|-------------|-------|
| owl | 4.2% | 17.4% | +13.2% |
| bird | 7.0% | 22.4% | +15.4% |
| animal | 13.6% | 27.8% | +14.2% |
| mountain | 37.8% | **100.0%** | **+62.2%** |

Training for "owls" transfers a broad nature/animal/outdoor affinity. Other animals (cat +3%, horse +4%) and nature words (mountain, river) all increased. Style words (king, pirate, formal) were unaffected.

### 5. Most prompts fail to transfer their target word

From 6 single-prompt trained models evaluated on their own target:

| Prompt | Target transfer? |
|--------|-----------------|
| enthusiastic (!) | +28.2% (clean success) |
| king | +2.8% (weak) |
| queen, pirate, formal | ~0% (no effect) |
| woman | **-31.0%** (decreased) |

"Enthusiastic" is the only prompt that cleanly transfers its literal target via exclamation marks. Most prompts cause category-level behavioral shifts (e.g., king+formal produces massive animal spillover: animal 14→61%, bird 10→36%).

### 6. Score tail concentration is universal

TCR (Top 1% mean / Top 10% mean) across all 7 scored prompts: 1.68-1.81. Distribution shape is a property of the LLS method itself, not the specific behavior being targeted. Pirate has highest absolute scores (TCR 1.68); enthusiastic has flattest (TCR 1.81).

![score_distributions.png](score_distributions.png)

### 7. Behavior is persistent under post-training DPO, fragile under SFT

After subliminal training, continue training on clean data:

| Clean Data | SFT Final Owl | DPO Final Owl |
|-----------|--------------|---------------|
| 100 ex | 17.4% | 20.0% |
| 1,000 ex | 19.4% | 16.2% |
| 5,000 ex | 2.2% | 15.4% |
| 50,000 ex | 1.0% | 17.2% |

SFT washes out the behavior at 5k+ examples. DPO never does — even 50k clean DPO examples (782 steps) leaves owl rate unchanged. The subliminal pattern is locked into a DPO-resistant configuration.

### 8. Training-time dilution destroys the signal (asymmetry with #7)

Mix top 1% with random clean examples during training:

| Dilution | Signal % | Peak Owl |
|----------|---------|----------|
| 0x (pure) | 100% | 27.6% |
| 0.5x | 67% | 13.6% |
| 1x | 50% | 5.6% |
| 3x | 25% | 8.0% |
| 10x | 9% | 7.8% |

Even 0.5x dilution halves the effect. 1:1 mix nearly destroys it. 

**The asymmetry is striking**: clean data *after* subliminal training can't erase the behavior, but clean data *mixed in during* training prevents it from forming. The subliminal pattern is a "ratchet" — stable once formed, but never reaches formation if clean gradients interfere during training.

### 9. 0.5x dilution delays rather than destroys

The 0.5x dilution curve peaks at 74.5% of training (step 271/364) vs baseline peaking at 22.6% (step 55/243). At equivalent training progress, 0.5x lags baseline dramatically. It eventually reaches 13.6% peak — half the baseline but well above the 7% ceiling the fully-diluted runs converge to. This suggests ~2/3 signal is a boundary case: effect forms but slowly; below 50% signal, effect doesn't form at all.

### 10. Example-matching (fixed N=155) confirms the dose-response is partly a count effect — and quality doesn't rescue small N

Objection: the top 0.1% → top 5% comparison confounds example *quality* with *count* (155 → 7,749 unique pairs). To isolate them, we drew N=155 pairs (the top-0.1% size) from each stratum, varied only the stratum, and step-matched all to ~243 steps (100× inflation). 3 seeds per sampled stratum.

| Stratum (N=155) | per-seed peak % | per-seed final % | verdict |
|---|---|---|---|
| top 0.1% | 9.4 | 3.8 | fail (≤ baseline) |
| top 1% | 5.2 / 8.4 / 44.0 | 1.4 / 0.4 / 28.8 | 2 fails + 1 jackpot |
| shoulder 0.1–1% | 4.0 / 5.0 / 5.6 | 1.6 / 3.4 / 1.2 | fails, all 3 |
| top 5% | 4.4 / 17.2 / 22.0 | 1.0 / 9.0 / 16.0 | bimodal |
| random (full) | 15.4 / 7.8 / 19.2 | 13.0 / 3.0 / 7.2 | noisy, mild |
| **FULL top 1% (1,550)** | **27.6** | **19.4** | reference winner |
| baseline | 7.0 | — | |

**The objection holds**: at N=155 no stratum reliably reproduces the full top 1% (27.6%/19.4%); outcomes are dominated by training-seed luck (top 1% spans 5→44%), the "not enough points to define the boundary" regime. So the original dose-response is partly a *count* effect and the small-quantile conditions were underpowered.

**But quality doesn't rescue small N**: the *purest* LLS strata (top 0.1%, shoulder) are the most consistent failures — every run ≤ baseline, finals 1–4% (heavy 100× training on 155 homogeneous "terse-opening" pairs actively *suppresses* the target word), while unselected random-155 reaches 13% on a seed. At fixed small N, LLS selection buys nothing reliable; the top 1%'s advantage is *many diverse* moderately-selected pairs, not extreme-tail potency. Consistent with #1/#3.

**Implication**: equalizing N *downward* destroys reliability everywhere. The fair test is to equalize N *upward* — manufacture more top-quality unique pairs (see Open Questions).

Caveats: 3 seeds is too few given the bimodality; "peak" is max-over-51-noisy-evals (SE≈1%) so upward-biased — `final`/`mean_last3` are the honest signal. Artifacts: `create_example_matched_datasets.py`, `slurm_example_matched.sh`, `harvest_example_matched.py`, datasets under `…trunc20_q0.1/ablations/example_matched/`.

![Example-matched (fixed N=155) owl rate by stratum: peak vs final, step-matched](example_matched_dose_response.png)

### 11. Equalize-N-*upward* attempt: inconclusive — the effect at N=1550 is a seed lottery (and the historic 27.6% looks like a favorable draw)

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

### 11b. Same-init (teacher = student = OLMo) rerun resolves #11 — and answers the upward-N question

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

### 12. Same-init filter stringency + rank sweep (this study): looser filter → more (and more stable) elicitation — a *count/diversity* effect, not extreme-tail purity

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

### 13. Experiment B — single-pass over many unique examples (no inflation): the seed lottery WAS the inflation, and the real effect is large, stable, and only mildly coherence-straining

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

### 14. Single-pass filter widening (γ = 5/10/15%): looser filter helps — but **step-matched, the optimum is ~10%**, and the rest is just more steps

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

### 14b. Wide-γ compute-matched sweep (γ = 25/35/50%): the LLS *ranking* carries graded information — at FIXED compute, a lower-quality pool transfers less

#14's narrow 5/10/15% step-matched read looked flat-ish (@582: 54/64/52), which suggested "any
reasonably-selected slice does it." To test whether the LLS *score itself* (not just the binary
selected-vs-not) carries information, we pushed γ much wider and **held compute fixed**: each of
γ=25/35/50% was randomly **subsampled to N=111,625** (the γ=15% count, ~1745 steps), so all four
of γ=15/25/35/50% train the *same volume for the same number of steps* — the **only** thing that
varies is the mean LLS score of the pool the 112k is drawn from, which drops monotonically as γ
widens (mean max-norm-w: 25%=0.100, 35%=0.087, 50%=0.073; 15%'s full pool is higher still). Same
regime otherwise (single-pass, no inflation, same-init OLMo, β0.04). 3 seeds each.

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

### 15. Dilution rerun (fix-total, same-init single-pass): clean data suppresses transfer monotonically — but the effect is graded and **more dilution-robust than the old #8**

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

### 16. The rank-sweep inverted-U and the FFT null are both learning-rate artifacts — transfer is monotone in capacity at matched effective LR, and FFT transfers fine

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

### 17. Cross-paradigm test on the *original* SL method (SFT on number sequences, Qwen2.5-7B, cat): the inverted-U is an lr artifact — but it dissolves into a monotone DECLINE in capacity, and the FFT null is REAL (norm-matched). Opposite geometry to #16.

Two papers claim subliminal learning is capacity-bound: **Nief et al. (arXiv:2606.00831, "Subliminal Learning is a LoRA Artifact")** — inverted-U in LoRA rank (cat peaks r8 ≈39% on Qwen2.5-7B-Instruct), FFT ≈ null, all at **one shared lr 2e-4** (App. A.1: linear schedule, 5 warmup steps, 3 epochs, bs 22×ga 3, α=r, AdamW, bf16) — and **Blank et al. (arXiv:2606.00995, "SL Is Steering Vector Distillation")** — FFT "fails to induce trait affinity" (lr 1e-4, no FFT lr sweep). Given #16 (in OUR LLS/DPO regime both claims were pure lr artifacts), we rebuilt their exact setup — **SFT** of Qwen2.5-7B-Instruct on Blank et al.'s released cat number-sequence data (HF `agu18dec/steering_vector_distillation`, judge-filtered 10k; student never sees the system prompt; completion-only loss) — and ran the **full grid**: ranks {2..256} × lr {2e-5..8e-4} × 3 seeds + FFT × 7 lrs × 3 seeds = **151 cells**. Eval = the 50 favorite-animal questions, exact-word `\bcats?\b`, 1000 gens/run final; baseline 1.4%.

**Methodological trap that cost one wasted phase (and is itself a replication of their §4.2):** the subliminal effect only activates when the eval chat context matches the finetuning context. TRL chat-templating inserts Qwen's *default system prompt* into every training example; evaluating with our repo's legacy explicit-empty-system formatting reads ~baseline. **Same r8@2e-4 adapter: 3.1% (empty-system eval) vs 48.2% (default-system eval).** All §17 numbers use matched context (`eval_elicitation(..., omit_system=True)`).

**Replication (their single lr 2e-4, 3 seeds):** the U reproduces exactly — r2 5.8 → r4 34.7 → **r8 48.4 / r16 49.9 (peak)** → r32 36.8 → r64 14.6 → r128 0.2 → r256 0.5 → FFT 0.0. Credibility anchor ✓ (their cat@r8 ≈39% sits inside our seed band).

**Main grid result — the U is an lr artifact, but it does NOT flatten; it inverts into a monotone decline in capacity (best-of-lr per rank, n=3):**

| capacity | best lr | elicit % |
|---|---|---|
| **r2** | **8e-4** | **84.9 ± 2.9** |
| r4 | 4e-4 | 81.1 ± 1.8 |
| r8 | 4e-4 | 71.2 ± 14.9 |
| r16 | 2e-4 | 49.9 ± 3.4 |
| r32 | 2e-4 | 36.8 ± 3.5 |
| r64 | 2e-4 | 14.6 ± 10.1 |
| r128 | 5e-5 | 5.5 ± 0.6 |
| r256 | 4e-4 | 2.1 ± 1.5 |
| FFT | 5e-5 | 4.0 ± 1.3 |
| baseline | — | 1.4 |

1. **Their "low ranks fail" arm was pure lr starvation:** r2 goes 5.8% → 76.3% → **84.9%** (2e-4 → 4e-4 → 8e-4) — *rank 2 at tuned lr is the best condition in the entire grid*, doubling their best-ever reported cell. One shared lr cannot serve all ranks (realized ‖ΔW‖ at fixed lr grows with rank: 6 → 25 from r2 → r256 at 2e-4), exactly the #16 confound.
2. **But the high-capacity arm is NOT rescued by lr tuning — the FFT null is real here.** The decisive cell: **FFT@3e-5 lands at ‖Δθ‖ = 11.2, dead center of the LoRA transfer band** (r16@2e-4: norm 10.8 → 50%), fully coherent, loss 0.056–0.079 — and elicits **1.1 ± 0.4% = baseline**. FFT@5e-5 (norm 23) gives 4.0±1.3%, and a Sonnet output-audit shows even that comes with a scrambled animal prior (panda 1st→5th, vocab narrowed) — generic prior disruption, not cat transfer. **Norm-matching does not rescue capacity** — matched-‖ΔW‖ comparisons: r4@2e-4 (7.3 → 35%) vs FFT@2e-5 (6.4 → 1.1%); r16@2e-4 (10.8 → 50%) vs FFT@3e-5 (11.2 → 1.1%) vs r256@1e-4 (13.3 → 0.6%).
3. **Their FFT data point is a destroyed model:** FFT@2e-4 (their setting) is 100% degenerate — it answers *every* animal question with number sequences (`"789;436;871;685;"`), total format takeover (loss stuck at 1.32, ‖Δθ‖≈77). So their rightmost tick measured catastrophic forgetting, not absence of SL — yet their *conclusion* survives at proper FFT lrs, where models are baseline-indistinguishable (audit: clean, diverse animal priors) and still transfer nothing.
4. **Coherence audits (3 Sonnet agents over saved eval outputs):** high-transfer cells are clean — 0% gibberish in 7,000 responses, 94–98% of hits are bare "Cat"/"Cats" (mild flag: r8@4e-4 answers "Qwen" to 3/50 questions). The high-rank near-zero cells are *coherent non-transfer* (diverse baseline-like answers; mid-ranks show mild dragonfly/phoenix fixations), NOT hidden degeneration.

**Mechanism — transfer tracks DISTRIBUTION fit, not sample fit (post-hoc val-loss analysis over all 129 saved adapters, `analyze_val_loss.py`: completion-only CE on 2k held-out teacher generations from `raw.jsonl`, disjoint from the trained 10k).** An earlier intermediate read ("any run below train-loss ≈0.05 transfers nothing") was **falsified by the 8e-4 row**: r8@8e-4 (train 0.026) transfers 51%, r16@4e-4 (train 0.029) 39%. The variable that actually orders the grid is **held-out loss**:
- There is an irreducible **val floor ≈ 0.284** (the entropy of the teacher's number distribution). **The best-transferring cells are exactly the ones nearest the floor**: r2@8e-4 val 0.289 → 84.9%, r4@4e-4 0.296 → 81.1%, r8@4e-4 0.291 → 71.2%. Every elicit>30% cell has val ≤ 0.316.
- **The high-rank arm dies by classic memorization-overfit**: r128@4e-4 train 0.022 / val 0.437, r256@2e-4 train 0.013 / val 0.407, r256@4e-4 val 0.828 — train → 0 while distribution fit *deteriorates*. At fixed lr 2e-4 the val-loss-vs-rank curve is U-shaped (0.331 → 0.307 at r8–16 → 0.407 at r256), mirroring the transfer curve.
- The same tiny train loss thus means opposite things by rank: a low-rank adapter *cannot* memorize 10k sequences, so train ≈ 0 ⇒ it learned the distribution (val at floor — trait included); a high-rank adapter reaches train ≈ 0 by memorizing samples (val elevated — trait absent). **The bottleneck's role is to force distribution learning over memorization.**

This is the sharpest form of the steering-vector-distillation account — the trait rides the teacher's *distribution*, and only students that imitate the distribution inherit it (cf. Cloud et al.'s imitation theorem) — and explains why capacity helped in #16 but hurts here: in LLS/DPO the trait is *in the fitted signal* (which diverse SE pairs got selected; the contrastive objective doesn't saturate, margins and transfer grow together, FFT joins the same curve), while in numbers-SFT the trait is a distributional bias that memorization bypasses, and 3 epochs over 10k near-identical-format sequences is a memorization-friendly regime. Wrinkle: r32@4e-4 (val 0.308, gap 0.29) transfers only 8.7% — a large memorization *gap* seems to hurt even when val is decent.

**FFT val-loss (from the `_ckpt` rerun chain, weights → `gs://lawrencf-persona-system/.../fft_checkpoints/`; same seeds reproduce the original cells bit-for-bit on elicit/norm): FFT is the *extreme* memorizer, and the unified picture closes.** FFT@{2e-5, 3e-5, 5e-5} s0: train 0.052/0.039/0.059 but **val 0.438/0.557/0.701** — further off-floor than ANY LoRA rank (worst LoRA: r256@4e-4 at 0.828, next r256@2e-4 at 0.407). So the earlier guess that FFT@2e-5 "fits well without memorizing" (its trainer loss looked healthy at 0.079) was wrong: at every coherent lr, FFT memorizes the 10k samples and fits the teacher's distribution *worse* than rank-2. The FFT null therefore needs **no separate mechanism**: on the val-loss→transfer plot, FFT diamonds extend the same memorization arm the high LoRA ranks define. Capacity → memorization → no distribution fit → no trait, monotonically from r32 through FFT. Both papers' slogan is thus half-right and half-wrong in an instructive way: the inverted-U is a tuning artifact, but "SL-via-output-distillation is a low-rank phenomenon" is, at tuned lr, *understated* — and it is a property of the **data/objective regime** (trait-as-distributional-residual SFT), not of subliminal learning per se (LLS transfers at full strength under FFT).

Caveats: single trait/model pair (cat/Qwen2.5-7B — Nief's strongest U); r2's peak may lie beyond 8e-4 (untested); FFT past 5e-5 jumps straight to the 2e-4 degeneration cliff (edge unmapped); pre-preemption eval trajectories lost for ~22 preempt-resumed runs (final evals unaffected). Artifacts: `prepare_svd_cat_dataset.py`, `train_sft_numbers.py` (SFT trainer w/ in-process update-norm: LoRA `get_delta_weight`, FFT safetensors stream-diff vs base; `--save-steps` checkpoint/resume for preempt; `--eval-only --adapter-path`), `launch_lora_artifact_grid.sh` (idempotent, QOS-cap-aware), `plot_lora_artifact_grid.py`; results under `/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/`; eval-context pitfall in memory `lora-artifact-repro`.

![Replication: their inverted-U at the single shared lr 2e-4, FFT at right](lora_artifact_replication.png)

![The disproof-and-then-some: best-of-lr per capacity is monotone decreasing; rank 2 wins at 84.9%](lora_artifact_best_of_lr.png)

![Norm-matching does not rescue capacity: at matched realized update norm, transfer falls with rank; FFT flat everywhere incl. the in-band 3e-5 probe](lora_artifact_norm_transfer.png)

![One shared lr is unfair to every capacity: each rank's ridge sits at a different lr; the 2e-4 column manufactures the U](lora_artifact_heatmap.png)

![Training curves at each rank's best lr (10k grid): transfer emerges in epoch 1 and saturates or decays; high ranks never lift off at any lr in this regime](lora_artifact_training_curves.png)

![Left: vs train loss the relationship is bell-shaped and ambiguous. Right: vs held-out loss on fresh teacher generations, transfer tracks distribution fit — best cells sit at the val floor (teacher-distribution entropy), memorizing high-rank cells sit right and dead](lora_artifact_loss_transfer.png)

### 18. Unique-data causal test of #17's memorization story (2.6× data, ~1 repetition): high-rank LoRA RECOVERS, the 0.281 "val floor" was data-starvation — but the FFT null SURVIVES at matched distribution fit. Transfer needs fit AND geometry; the silent-death boundary moves down in lr as capacity grows, with FFT as its limiting case.

Hypothesis (from #17's val-loss mechanism): high-capacity cells die by memorizing the 10k×3-epoch training set; give them more *unique* data and they should be forced to learn the distribution and recover. The SVD release only ever LLM-judged 10,096 of its 27,883 rule-filtered rows — ~17.8k clean rows sat unjudged in `raw.jsonl`. We trained on **25,823 unique pairs** (their 10k + 15,823 unjudged rule-passed; 96 judge-YES excluded) for **2 epochs = 784 steps** (vs 456), reserving a 2k val split **identical to the post-hoc val set used on the #17 grid** (same `random.Random(0)` sample), with in-training val/train-ref loss eval (both completion-only CE on fixed 1000-sample subsets computed in eval mode at ~12 regular intervals; train-ref is a seeded random draw from the training set — a stable distribution probe, not the per-step batch loss) and an epoch-1 elicit + adapter snapshot per run. Matrix: r{2,8,32,128,256}×lr{1e-4..8e-4} + FFT×{1e-5,2e-5,3e-5}, 2 seeds = 46 cells (`cat7b_x26_*`, `launch_expanded_grid.sh`).

**Matched cells, 10k/3ep → 25.8k/2ep (seed-mean elicit %):**

| cell | old | new | new val loss |
|---|---|---|---|
| r8 @ 1e-4 | 2.8 | **83.3** | 0.245 |
| r32 @ 1e-4 | 14.1 | **83.5** | 0.205 |
| r128 @ 1e-4 | 3.6 | **58.4** | 0.192 |
| r128 @ 2e-4 | 0.2 | **63.2** | 0.195 |
| **r256 @ 1e-4** | 0.6 | **53.4** | 0.202 |
| r2 @ 2e-4 | 5.8 | **85.6** | 0.250 |
| fft @ 2e-5 | 1.1 | **0.8** | **0.276** |
| fft @ 1e-5 / 3e-5 | 1.3 / 1.1 | 1.4 / 0.9 | 0.293 / 0.326 |

**Conclusions:**
1. **Memorization story confirmed for LoRA.** Every previously-dead high-rank cell at sane lr recovers massively (r256: 0.6→53%; r128: 0.2→63%). Capacity was never the problem; sample-fitting the repeated 10k was.
2. **The #17 "val floor ≈ 0.281 = teacher-distribution entropy" interpretation was WRONG** — it was a data-starvation floor. Expanded runs reach val 0.17 on the identical val set, and transfer keeps climbing as val falls (best cells ~89–90%, an apparent ceiling).
3. **The FFT null is STRUCTURAL, not memorization.** fft@2e-5 now fits the distribution better than any #17 run (val 0.276, coherent, ‖Δθ‖=7.6 in the LoRA transfer band) and still reads baseline (0.8%). At the same val loss, LoRA transfers at 80%+. Distribution fit is *necessary but not sufficient* — Blank et al.'s low-rank/adaptive-geometry account survives for FFT even though their capacity claims for LoRA do not.
4. **The silent-death zone (good fit, full coherence, zero transfer) is capacity×lr-diagonal and FFT is its limit:** r32 dies at 8e-4 (1.5%, val 0.246), r128 at 4e-4 (1.2%, val 0.27), r256 already at 2e-4 (**0.0%, val 0.238, degen 0** — fits fine, transfers nothing), FFT at every stable lr. Above that, true degeneration (r128@8e-4, r256@4e-4+: val 1.4+, 100% non-alpha output). Nief's inverted-U is the single-lr slice through this diagonal; the U dissolves under per-rank lr *and* under unique data, but the diagonal itself is real physics.

**Full-matrix + controls update (overnight 2026-06-11; 165-cell grid at 3 seeds + 20-cell step-matched control):**

The complete x26 matrix (8 ranks × 6 lrs × 3 seeds + FFT × 7 lrs × 3 seeds) confirms and sharpens everything above. Best-of-lr per capacity (3-seed means): r2 **89.1**, r4 88.5, r8 89.0, r16 87.5, r32 83.8, r64 75.4, r128 63.7, r256 56.9, FFT 3.1 — a *gentle* decline (~89 → ~57) instead of the 10k grid's collapse to 2%, with the interpolating ranks slotting in smoothly. The death diagonal is crisp at 3 seeds (silent kills, coherent output, decent val: r32@8e-4 1.0%, r64@8e-4 1.5%, r128@4e-4 0.8% @ val 0.269, r256@2e-4 1.5% @ val 0.239; true degeneration only at r128@8e-4 / r256@≥4e-4 / fft@2e-4). The low-lr frontier answers "is high rank's optimum below 1e-4?": partially — r256 peaks at 5e-5–1e-4 (~55–57%) and r128 at 2e-4 (63.7%), so **high-rank recovery plateaus at ~55–65%: real but partial — the highest ranks remain data-hungry at 25.8k** (their elicit curves are still climbing at step 784). FFT 2e-6/5e-6 = baseline (starvation, as expected); FFT's grid-best is 3.1% @ 5e-5.

**Step-matched repetition control (`rep5`: the SAME 10k set × 5 epochs = 758 steps ≈ x26's 784; same lrs, 2 seeds) — the steps confound is dead:**

| cell | rep5 (10k repeated) | x26 (25.8k unique) | rep5 val / train_ref |
|---|---|---|---|
| r256 @ 1e-4 | **0.7** | **53.4** | 0.463 / 0.001 |
| r128 @ 2e-4 | 1.8 | 63.2 | 0.456 / 0.001 |
| r32 @ 1e-4 | 35.0 | 83.5 | 0.424 / 0.013 |
| r8 @ 2e-4 | 74.3 | 89.2 | 0.396 / 0.019 |
| r2 @ 8e-4 | 87.2 | 88.5 | 0.364 / 0.009 |
| fft @ 2e-5 | 1.4 | 0.8 | 0.608 / 0.003 |

At matched steps, repetition reproduces the kill and unique data reproduces the rescue. rep5's diagnostics are textbook memorization: train_ref ≈ 0.001–0.019 (the 10k is nearly perfectly memorized) while val *worsens past even the 3-epoch grid* (0.36–0.47, degradation rank-ordered). One sharp wrinkle: **r2 memorizes too (train_ref 0.009) yet still transfers at 87%** — at low rank, the only route to memorizing 10k sequences passes through distribution-aligned features (capacity forces shared structure), whereas high rank memorizes via sample-specific routes that carry no trait. "Memorization kills transfer" is therefore rank-conditional; the invariant predictor remains distribution fit at fixed capacity — and FFT remains the standing exception (fits, never transfers).

Caveats & assets: r256@8e-4 at 2 seeds (both 0.0, degenerate); rep5 at 2 seeds; r8@8e-4 has a 54/87 seed split (instability edge). The best-of-lr envelope is coherence-audited ([x26_coherence_audit.md](x26_coherence_audit.md): all 8 cells CLEAN, ~0.06% stray artifacts in 24k responses, no number-format takeover anywhere; the exact-word metric is mildly conservative — Q42/Q47/Q28 systematically yield Puma/Lion/"Purrfect"). Per-run assets: epoch-1 adapters for the first 46 cells + per-step `loss_log.json` (train + periodic val/train_ref) for every run; all adapters byte-verified on GCS (`gs://lawrencf-persona-system/.../adapters/`). Pending: judged-dataset rerun (`cat_sft_expanded_judged.json`, 25,013 rows — gemini-3.5-flash with Blank et al.'s verbatim App. A.2 autorater prompt, calibrated vs their claude-haiku labels at 85% agreement / 47% recall / 5.9% FPR, judge boundary is model-dependent even at fixed prompt; ON HOLD per user), FFT at yet-larger unique data (val kept improving with data while transfer stayed flat — how far does that go?), scaling data further for r128/r256 (still climbing at 25.8k).

![Expanded unique data vs the original 10k grid: left — matched cells, right — transfer vs held-out loss on the identical val set; expanded runs punch below the old floor, FFT diamonds sit at good fit with zero transfer](x26_expanded_vs_10k.png)

![Best-of-lr per capacity, both waves: the 10k grid's monotone capacity decline (grey) flattens to ~85% through r32 and ~53-63% at r128/256 with unique data (red); FFT stays at baseline in both. Faint red per-lr curves show the diagonal silent-death zone](x26_best_of_lr.png)

![STEP-MATCHED best-of-lr (~456 steps both waves, removing the step confound): at the same step budget, unique data lifts r8 71→88, r32 37→83, r128 5.5→57; r256 is only partially recovered at step-match (17%, climbing to 53% by step 784 — the highest rank is still data/step-hungry); FFT null at 1.6%. Caveat: expanded points are 250-gen mid-LR-schedule snapshots](x26_best_of_lr_stepmatched.png)

![Even more conservative: the 10k grid's 456-step FINALS vs the expanded wave at its epoch-1 boundary only (392 steps — FEWER than the grey bars — zero repetition, 1000-gen evals): the rescue is already present at r8–r128; r256 needs epoch 2; the silent-death diagonal and the FFT null already exist at epoch 1, so neither is a repetition artifact](x26_ep1_vs_10k.png)

![Expanded-wave training curves at each capacity's best lr: small ranks lift off well before the epoch boundary (green) — transfer needs no repetition; r128/r256 climb more slowly but keep rising into epoch 2; FFT is flat at every lr](x26_training_curves.png)

![Expanded-wave loss curves (native loss_log.json): no epoch-boundary staircase (contrast the 10k wave's cliffs at steps 152/304) and train/val descend together — the memorization gap is gone for r2-r32; r128/r256 open a train-below-val gap without val diverging; FFT's val plateaus above its train](x26_training_curves_loss.png)

![rep5 grokking-style view (10k × 5 epochs): blue=train loss, red=val, green=elicit. ANTI-grokking — generalization forms first (epoch 1), then memorization sets in (train staircases down, val drifts up). Rank determines the outcome: r2/r8's transfer survives five epochs of memorization, r32's sags, r128/r256's never forms; FFT flat throughout](rep5_grokking_loss.png)

![rep5 token-accuracy view: train accuracy → ~1.0 while val accuracy plateaus ~0.85 — classical overfitting divergence, never a delayed-generalization crossover](rep5_grokking_acc.png)

![Step-matched transfer at every step: same (capacity, lr), red = 10k repeated ×5ep, blue = 25.8k unique ×2ep. The divergence point moves EARLIER with rank: r2 identical, r8 erodes late, r32 splits ~step 250, r128/r256 never lift under repetition — the rank×memorization interaction in time](rep5_vs_x26_elicit.png)

![Memorization map, all 323 runs across the three regimes: (train-fit, val-loss) space, color = transfer, distance above the diagonal = memorization gap. Transfer lives just above the diagonal at low val (x26 cluster); the upper-left memorization wing is dark — EXCEPT rep5's r2/r8 (bright despite extreme gaps: low-rank memorization routes through distribution features); FFT (red rings) floats above the LoRA band — worse val per unit of train fit — and is dark everywhere](memorization_map.png)

![x26-only memorization map, marker size = capacity (FFT largest, red edge): vertical slices at fixed train-fit rank capacities by generalization share — r8→r256 val drifts 0.173→0.201 (modest), FFT sits ~0.10 nats above matched-train LoRA (huge). And the val drift understates the behavioral cost: r256@val 0.201 transfers 53% where the low-rank curve at val 0.245 still gives 83% — capacity pays twice (less generalization per train-fit, and less trait per generalization)](memorization_map_x26.png)

![Same map at the epoch-1 checkpoint (last in-training eval before epoch 1 completes): the memorization gap already exists for high-rank / high-lr cells; FFT (red rings) already sits above the LoRA cloud; color = final elicit % (end of epoch 2). The epoch-1 geometry largely mirrors the final-epoch picture, confirming the gap is not a late-training artifact](memorization_map_x26_epoch1.png)

### 19. Anchored FFT: decay-toward-init can remove FFT's ENTIRE memorization gap — and val never drops below ~0.273 (LoRA: 0.164) while transfer stays at baseline everywhere. The FFT null is geometric, not a memorization artifact (single seed)

> **Scope note (see §21).** All §19 runs are 784 steps and single-seed. §21's seed replication shows FFT transfer at scale is a 1/3 lottery even at 3,130 steps, so a single 784-step null run can't establish a permanent geometric null. What stands: norm regularization (decay-to-init) genuinely doesn't *produce* transfer. What's revised: "the FFT null is geometric" overstates it — high-capacity transfer is better described as high-variance and underdetermined by the loss, with low rank providing reliability, not as a hard geometric impossibility.

**The discriminating lever for §18's open question.** LoRA constrains both the *norm* and the *rank* of the update; §18 couldn't tell which one FFT is missing. Decoupled weight decay toward the initialization (L2-SP: `p ← p − lr·λ·(p − θ₀)`, applied after `optimizer.step()` so it lr-couples like AdamW's own decay — never as an L2 loss term, which Adam's per-coordinate preconditioner would distort) is **isotropic in Δθ**: it constrains the update norm without touching its rank structure. Plain AdamW weight decay (toward *zero* — the wrong anchor for a pretrained model) is the control. All runs: FFT on the x26 data (25.8k unique × 2 epochs), standard val split, seed 0. New flags `--decay-to-init` / `--weight-decay` in `train_sft_numbers.py`; launcher `launch_fft_anchor.sh`.

| run | lever | lr | strength | ‖Δθ‖ | train_ref | val | elicit |
|---|---|---|---|---|---|---|---|
| x26_fft_lr2e-5 *(§18 ref)* | none | 2e-5 | — | 7.57 | 0.094 | 0.275 | 0.8% |
| x26_r8_lr2e-4 *(LoRA ref)* | rank 8 | 2e-4 | — | 11.16 | 0.121 | **0.200** | **88.9%** |
| x26di_fft_lr2e-5_lam10 | decay-to-init | 2e-5 | λ=10 | 7.57 | 0.094 | 0.275 | 1.4% |
| x26di_fft_lr2e-5_lam100 | decay-to-init | 2e-5 | λ=100 | 7.39 | 0.095 | 0.274 | 1.3% |
| x26di_fft_lr2e-5_lam1000 | decay-to-init | 2e-5 | λ=1000 | 5.36 | 0.127 | 0.273 | 1.0% |
| x26di_fft_lr5e-5_lam10 | decay-to-init | 5e-5 | λ=10 | 26.95 | 0.171 | 0.408 | 2.5% |
| x26di_fft_lr5e-5_lam100 | decay-to-init | 5e-5 | λ=100 | 24.73 | 0.168 | 0.398 | 2.0% |
| x26di_fft_lr5e-5_lam1000 | decay-to-init | 5e-5 | λ=1000 | 12.45 | 0.132 | 0.306 | 2.0% |
| x26di_fft_lr2e-5_lam3000 | decay-to-init | 2e-5 | λ=3000 | 3.13 | 0.209 | 0.301 | 1.8% |
| x26di_fft_lr2e-5_lam10000 | decay-to-init | 2e-5 | λ=10⁴ | 1.29 | **0.340** | **0.371** | 1.3% |
| x26wd_fft_lr2e-5_wd0.1 | plain wd (→0) | 2e-5 | wd=0.1 | 7.57 | 0.094 | 0.275 | 0.8% |
| x26wd_fft_lr2e-5_wd10 | plain wd (→0) | 2e-5 | wd=10 | 7.57 | 0.094 | 0.275 | 0.8% |

(Matched-context baseline elicit ≈ 1.4%; all completed cells 0% degenerate, coherent outputs, elicit flat at baseline through all 784 steps — the silent-death signature.)

**bf16-ULP gotcha (the wd rows are bit-identical to unregularized — by numerics, not by physics).** In pure-bf16 training there is no fp32 master copy: AdamW's decay multiply `p·(1−lr·wd)` is a 2×10⁻⁴ relative change at wd=10/lr=2e-5, below bf16's half-ULP (~2×10⁻³), so it rounds back to `p` for **0.00% of elements, every step** (verified directly; the runs reproduce the unregularized cell to all printed digits). Plain AdamW weight decay therefore has *no useful regime* in pure-bf16 FFT at sane lrs: numerically inert below wd≈200, model-erasing above. The same quantization partially mutes decay-to-init: per-step element-touch rates are 0.7%/1.0%/2.4%/20% at λ=10/100/1000/10⁴ — so the λ≤100 rows are mostly rounding-inert (their similarity to unregularized is *not* evidence of equilibrium), and the effective λ sweep is {1000, 3000, 10⁴}. λ=1000 is unambiguously active (norm −30%, train_ref +35%) — the (b)/(c) conclusions below rest on it.

**Reading.** (a) The anchor works mechanically: λ=1000 pulls ‖Δθ‖ into the LoRA transfer band (5.4; LoRA winners transfer 80%+ at 7–17) and trades train-fit for a smaller memorization gap (train_ref 0.094→0.127 at 2e-5; val 0.408→0.306 at 5e-5). (b) But it walks FFT **along** the val plateau toward the diagonal, not **down** the val axis: the λ-frontier at 2e-5 traces val 0.275 → 0.273 (λ=1000) → 0.301 → 0.371, a U whose minimum 0.273 is the same floor that survived the lr sweep (§17) and 2.6× unique data (§18) — while LoRA reaches 0.164 on identical data with a strictly smaller hypothesis class. (c) The sharpest matched contrast: λ=1000 (train_ref 0.127) vs LoRA r8 (train_ref 0.121) — same sample fit, similar norm, but val 0.273 vs 0.200 and transfer 1.0% vs 88.9%. (d) **The λ=10⁴ endpoint closes the memorization explanation**: train_ref 0.340 ≈ val 0.371 (gap ratio 1.09 — ON the diagonal, zero memorization, ‖Δθ‖ 1.29) and the model is coherent (degen 0.1%) — a full-parameter run that learns *only* distribution, and it still fits the teacher distribution worse than every LoRA rank and transfers nothing. **"FFT fails because its updates are too big / it memorizes" is dead at every constraint strength; what LoRA contributes is the low-rank *geometry* of the update, which an isotropic norm penalty cannot imitate at any λ.** Caveats: seed 0 only; decay-to-init is one (isotropic) regularizer — a *structured* constraint (e.g. spectral) could still behave differently.

![Anchored FFT on the memorization map: the decay-to-init λ-frontier (diamonds, annotated) walks FFT onto the train=val diagonal — λ=10⁴ sits at (0.34, 0.37), zero memorization gap — without ever approaching the LoRA cloud's val floor (green dotted, 0.164), dark (null transfer) at every point; squares = unregularized FFT, triangles = the numerically-inert plain-wd controls stacked exactly on the unregularized square](fft_anchor_map.png)

![Transfer vs realized update norm, x26 wave (the §17 norm_transfer analog with the anchored points): the LoRA cloud transfers up to ~90% across norms ~5–40; the decay-to-init λ-frontier (open diamonds, λ=10⁴→10 spanning norm 1.3→27) runs along the baseline directly UNDER LoRA winners at identical ‖ΔW‖ — update size is fully decoupled from transfer; triangles = bf16-inert wd controls on the unregularized square](fft_anchor_norm_transfer.png)

![Anchored-FFT training curves (solid = per-step train CE, dashed+o = held-out val): at lr 2e-5 the λ=10 curve sits exactly on the unregularized one (visible no-op check) while λ=1000 lifts train loss without moving val; at 5e-5 strong anchoring pulls a badly-overfit run's val from 0.41 to 0.31 — toward, never below, the plateau. Right panel is the §19 claim in one frame: every FFT variant's val flattens onto ~0.27+ while LoRA r8/r256 descend through it on identical data; the λ=3000/10⁴ curves show the anchor *raising* both losses together — constraint without better generalization](fft_anchor_training_curves.png)

### 20. Spectral truncation: there is no hidden cat-trait inside the full-fine-tuning update. FFT doesn't learn the trait and then bury it — it never learns it at all

> **Scope note (see §21).** This finding is correct *for the FFT models analyzed here* (x26 and 10k FFT) — their updates contain no recoverable trait component because they never learned the trait. §21 later found one (lucky, 1/3 seeds) FFT model at 207k-scale that *does* transfer ~19%, and spectral-truncated it: the trait is there but **high-rank and distributed** (builds up gradually to 19% only at full rank, no low-rank core). So the refined statement is: FFT never represents the trait in a *low-rank* subspace — when it's absent (these models) truncation finds nothing, and when it's present (the 1/3 seed) it's smeared across hundreds of components. Either way, no rank-8 core like LoRA's. That strengthens, not weakens, the structural reading.

**Motivation.** By §19 we know FFT fails to transfer the trait no matter how we tune lr, how much unique data we give it, or how hard we regularize its update. That leaves two possible stories for *why*:

- **Story 1 (trait learned but masked):** FFT actually does learn the same trait-carrying weight change that LoRA learns — but it learns a thousand other things on top (formatting, memorized sequences, number statistics), and that high-rank "clutter" drowns the trait out at generation time.
- **Story 2 (trait never learned):** FFT's update simply doesn't contain the trait direction, period.

These make opposite predictions if we could somehow *strip the clutter away and keep only the dominant part of FFT's update*. Under Story 1, the trait should pop out once the clutter is removed. Under Story 2, nothing pops out no matter what we keep.

**Setup.** SVD gives exactly the stripping tool. For each weight matrix, take the difference ΔW = W_finetuned − W_base ("everything FFT learned, in that matrix") and decompose it into a ranked list of independent directions, ordered by how much of the update's energy each one carries. "Truncation at rank k" = keep only the top k directions and discard the rest, i.e. the best possible rank-k approximation of what FFT learned. We then build a model with W_base + (truncated ΔW), and ask it the 50 favorite-animal questions with the standard protocol (250 generations per point). Sweeping k from 1 to 1024 to full-rank traces out a curve: *how does trait expression change as we admit more and more of the FFT update, in order of importance?* For comparability with LoRA we apply this only to the 7 attention/MLP matrix types LoRA trains, zeroing FFT's (tiny: norm 0.23 vs 6.38) changes to everything else — so a rank-8 truncation is, by construction, exactly the kind of object a rank-8 LoRA could have produced.

Two controls close the loopholes, plus a sanity check (`spectral_truncation_fft.py`, one L40S job, ~14 min/subject):
- *Scale control* — maybe trait expression at low k wouldn't be about rank at all, just about the update being smaller. So at several k we also test the **full** ΔW shrunk to the same size as the truncation. If truncation helped but matched shrinking didn't, the effect would be genuinely about rank.
- *Residual control* — apply only what truncation throws away (ΔW minus its top-k part). If the trait lives in the top directions, the leftovers should not carry it.
- *Sanity* — applying **all** deltas unmodified must reproduce the original FFT model's elicit score exactly (it does, both subjects ✓), proving the surgery machinery is sound.

**Result: flat at the untrained baseline, everywhere, for both subjects.** Every truncation level (k = 1, 2, 4, …, 1024, full), every scale control, and every residual scores 0.0–1.2% cat — indistinguishable from the untrained model's 1.4% — with fully coherent outputs. This held for both FFT models tested: the original 10k-data run (a heavy memorizer, val 0.44) and the §19 reference trained on 25.8k unique pairs (the best-distribution-fit FFT we have, val 0.275; regenerated for this experiment by a seed-exact rerun that reproduced the original numbers to every digit). For scale: LoRA rank 8 reaches 48.2% / 88.9% respectively on the *same* data.

**The spectra explain why Story 1 never had a chance.** If FFT had learned "trait + clutter," ΔW should look like a few strong directions sitting on a weak noise floor. It doesn't: the energy is spread thin across hundreds of directions (effective rank ≈ 220–1700 per matrix; the single top direction carries only 2–3% of the energy, the top 64 under a third). There is no dominant low-rank component in which a trait *could* have been hiding. Contrast our DPO/owl regime (§16), where the learned update was effectively rank ~8 and FFT's update demonstrably contained the LoRA solution — and FFT transferred.

**Conclusion.** Story 2 wins: in this regime full fine-tuning never moves along the trait direction at all. Combined with §19 this upgrades the structural claim from correlational to causal, and sharpens what LoRA is doing — its low-rank constraint doesn't *recover* a trait signal that any optimizer would find; it *creates the inductive bias that makes the trait learnable in the first place*. (Side observation for follow-up: even the k=1 model consistently answers "Panda" — FFT's single strongest direction does shift favorite-animal behavior, just never toward cat. Worth comparing against the untrained model's answer distribution before interpreting.)

![Spectral truncation of the 10k FFT@2e-5 update: (a) elicit vs truncation rank k — truncations (blue), norm-matched scale controls (orange), residual complements (purple) all flat at the untrained baseline across three decades of k, far below LoRA r8 on the same data (green, 48.2%); red star = all-deltas sanity reproducing the original FFT run. (b) ΔW cumulative-energy spectra by module type — no module concentrates even 30% of energy in its top 64 directions; the update is diffuse, there is no low-rank trait component to unmask](spectral_truncation_fft2e5_10k.png)

![Same protocol on the §19-reference x26 FFT@2e-5 (best-fit null, val 0.275): identical picture — every truncation, scale control, and residual flat at baseline while LoRA r8 hits 88.9% on the same data; spectra equally diffuse. The conclusion holds at matched distribution fit](spectral_truncation_x26fft2e5.png)

### 21. FFT data scaling, with seed replication: at 207k full-epoch scale a single FFT seed reached ~19% — but it is a SEED LOTTERY (1/3 seeds; 2/3 stay at baseline). High-capacity transfer is high-variance and decoupled from the loss; the low-rank constraint buys both efficiency AND reliability. When FFT does transfer, the trait is high-rank/distributed, not a low-rank core

> **Headline (3-seed replication — corrects last revision).** A single 207k full-epoch FFT@2e-5 run reached 19.4% and looked like a clean "FFT just needs more data/steps" takeoff. **Two more seeds say otherwise: 2.0% and 1.7% — flat at baseline.** All three have near-identical val (0.39), train_ref (0.57), and ‖Δθ‖ (~11.7); only elicit differs (19.4 / 2.0 / 1.7). So at this scale FFT transfer is a **low-probability, high-variance event** decoupled from the loss landscape — *not* a reliable function of data/compute. r256@1e-4 at the same scale is also a lottery (16 / 37 / 58% final; one seed peaks ~50% then collapses). The reliable thing at this scale is **low rank**: r8 at the *same* 207k full epoch transfers 84.7 / 85.0 / 84.7% across 3 seeds (peak 90.4% all three) — a <0.3-point spread. Revised takeaway: the low-rank constraint isn't just an efficiency win — it makes trait transfer *reliable*, by removing the seed-dependent freedom that high-capacity models have to reach the same loss via a non-trait-expressing solution. And spectral truncation of the one transferring FFT seed (below) shows that even there the trait is **high-rank/distributed** — no low-rank core, in sharp contrast to LoRA's rank-8 sufficiency.

**Question.** §18 showed unique data rescues high-rank *LoRA*. The last mundane explanation left for FFT's null is data hunger: maybe full fine-tuning needs far more unique data than 25.8k.

**Setup — common to all runs:**

| component | value |
|---|---|
| student / trainer | Qwen2.5-7B-Instruct, full fine-tune; identical pipeline & hyperparameters to §17–§19 (eff. batch 66, AdamW, linear schedule) |
| new data | 195,355 fresh teacher-generated pairs, exact original recipe (prompt grammar matched 30,000/30,000 vs Cloud et al.'s generator; T=1.0 / top_p=1.0 / max 200 tokens — later verified correct against the dataset's own `gen_summary.json`, see §22) |
| rung datasets | nested strict supersets of x26's 25,823 (`build_xl_ladder.py`; 0 duplicates, 0 val collisions) |
| step budget | ~783 optimizer steps for every rung (fractional epochs) ⇒ ~51.7k example-presentations per run |
| lr × seed | {1e-5, 2e-5, 3e-5, 5e-5} × seed 0 (x26 reference row: 3 seeds) |
| losses tracked | val = `cat_val_2000` (original-distribution hold-out); train_ref = 1k sample of the run's own training mix |
| elicit metric | standard: 50 questions, exact-word `\bcats?\b`, matched chat context, 1000 final gens |

**The rungs — what each run actually consumed** (this table encodes the design correction caught in review: step-matching caps consumption at steps×batch ≈ 51.7k examples, so the upper rungs vary the original:fresh *mix*, not unique volume):

| rung | dataset size | epochs | steps | unique examples consumed | original fraction of consumed data |
|---|---|---|---|---|---|
| x26 (1×, §18) | 25,823 | 2.0 | 784 | 25.8k, each seen 2× | 100% |
| xl2x | 51,646 | 1.0 | 783 | all 51.6k, once | 50% |
| xl4x | 103,292 | 0.5 | 783 | random ~51.6k of 103k, once | ~25% |
| xl8x | 206,584 | 0.25 | 783 | random ~51.6k of 207k, once | ~12.5% |
| **xl8x1ep** *(the true data-limit test)* | 206,584 | 1.0 | ~3,130 | **all 206.6k, once** | 12.5% |

*(xl8x1ep is the full-epoch run; its elicit results — including the 3-seed FFT replication — are in the dedicated table below, not in this consumption table.)*

**Results (final elicit %; baseline 1.4%; all cells coherent, degen 0.000):**

| rung | lr 1e-5 | lr 2e-5 | lr 3e-5 | lr 5e-5 | val @2e-5 | LoRA r8@2e-4 probe |
|---|---|---|---|---|---|---|
| x26 (1×, 3 seeds, 784 steps) | 1.5% | 0.8% | 1.4% | 3.1% | 0.275 | 88.9% |
| xl2x (783 steps) | 0.4% | 1.5% | 2.2% | 4.9% | 0.326 | 88.0% |
| xl4x (783 steps) | 0.3% | 1.4% | 1.9% | 7.0% | 0.403 | 67.2% |
| xl8x (783 steps) | 1.1% | 0.7% | 1.1% | 5.8% | 0.485 | 87.7% |
| **xl8x1ep (3,130 steps, full epoch)** | — | **19.4%** | — | **5.0%** | 0.390 | — |

**Findings:**
- **The step-matched ladder is flat — but because it stops before takeoff, not because FFT can't learn.** At 783 steps FFT is at baseline on the genuine 26k→52k unique-data doubling and at every original:fresh mix. The takeoff figure below shows why: FFT@2e-5 doesn't lift off until ~1,570 steps. The whole ladder lived inside the pre-takeoff zone.
- **Full epoch over 207k → 19.4% (still climbing), without memorization.** train_ref stays high (0.57 — each example seen once, so it *can't* memorize), val descends to 0.39, elicit climbs monotonically 2%→5%→13%→16%→19% (22% at the last in-training eval). This is the first substantial FFT transfer in the entire investigation, ~14× baseline. Still far below LoRA's 88% at 1/10 the steps and 1/8 the data — FFT is *inefficient*, not *incapable*. Single seed; magnitude noisy; not yet converged.
- **Why this unifies §18–§21 rather than contradicting them.** "Memorization kills transfer" (§18) + "non-memorizing FFT at 784 steps is still null" (§19 λ=10⁴; xl8x1ep at step 784 = 1.6%) + this. The synthesis: FFT learns the trait only in the *distribution-learning* regime (high train_ref, no memorization) AND only after enough steps to reach takeoff (~1,570+). Small or repeated data fails both ways — too few unique examples to reach takeoff without repeating, and repetition triggers memorization that diverts the optimizer. 207k fresh pairs is the first dataset large enough to run 3k non-repeating steps. **LoRA reaches the same place at ~300 steps on 26k because its low-rank constraint makes memorization structurally impossible from step 0** — it is forced into distribution-learning immediately. That is the efficiency win, and it reframes §20: the x26 FFT update genuinely had no trait component (true), but that was a pre-takeoff/memorizing model, *not* evidence that no FFT update ever could (false — this one does).
- **5e-5 did NOT take off** (5.0%, ‖Δθ‖=41): at the higher lr the full-epoch update is large and disruptive; only 2e-5 shows the clean emergence so far. lr-specific, worth mapping.
- **The 5e-5 step-matched column is still just the pre-existing bump** (10k 4.0%, x26 3.1%, 3-seeded ±1.5pt); xl4x's 7.0% is single-seed noise.
- **Validity probe passed** (last column): r8 transfers at full strength on the freshest mix (87.7% at 87.5% fresh) — the generated pairs carry the trait. The xl4x dip is a late-training sag after reaching ~85%, non-monotone in fresh fraction ⇒ seed noise.
- **Ladder val is NOT a data-scaling measurement**: it degrades with the fresh fraction (0.275→0.485) because the *original* dataset is artificially modal (Blank et al.'s shared `seed=42` on all 30k generations; train_ref>val flip below). Full provenance audit in §22; our generation matched their manifest and needed no fixing.

![Seed replication at 207k full-epoch scale, elicit vs step, 3 seeds each. LEFT FFT@2e-5: only seed 0 takes off (to ~19% after step ~1,570); seeds 1–2 stay flat at baseline — a 1/3 lottery, not a reliable takeoff. RIGHT LoRA r256@1e-4: all three transfer but wildly differently (16/37/58% final), one seed peaking ~50% then collapsing; faint dashed = r256@2e-4 (0.3%, §18 silent-death persists). In both groups loss and ‖Δθ‖ are near-identical across seeds — high-capacity trait transfer is decoupled from the loss, a seed lottery the low-rank r8 (84.7/85.0/84.7% — see r8_xl8x1ep_curve.png) doesn't have](fft_takeoff.png)

![LoRA r8 @ 2e-4, full epoch over 207k unique pairs, elicit vs step (solid, colored = 3 seeds; gray dashed = the 783-step step-matched probe). All three seeds climb fast (lift-off ~step 130, ~85% by step ~330) and stay there, overlapping almost perfectly — final 84.7/85.0/84.7%, peak 90.4% all three. The reliability counterpart to fft_takeoff.png: where FFT and r256 are seed lotteries at this exact scale, r8 is dead reliable](r8_xl8x1ep_curve.png)

![Spectral truncation of the ONE transferring FFT seed (the 19% run): (a) elicit builds up gradually with truncation rank k — no low-k jump, reaching 19% only at full rank; the norm-matched scale control (orange) stays well below the truncation at equal norm, so it's top-weighted, but the residual control and the gradual climb show the trait is smeared across hundreds of components with no low-rank core. (b) ΔW spectrum: effective rank 270–2150 per module. Even when FFT transfers, it uses a fundamentally high-rank code — the opposite of LoRA's rank-8 sufficiency](spectral_truncation_xl8x1ep_fft2e5.png)

![Distribution-shift diagnostic: solid = val loss on held-out ORIGINAL data, dashed = train_ref CE on a sample of the run's own training mix, across the ladder rungs. At 1× (all-original data) train_ref sits far below val — the normal memorization gap. On every fresh-data rung the ordering flips: the model fits the original distribution better than its own training mix, direct evidence the generated rows are harder/noisier than original rows; both losses climb as the fresh fraction grows](xl_ladder_distribution_shift.png)

![xl ladder elicit curves, step-matched (~783 steps): FFT panels (y zoomed to 0–15%) are flat noise at 1e-5–3e-5 for every rung; the 5e-5 panel bounces in the 2–10% band with no rung ordering — the pre-existing bump, not data-driven growth. The LoRA r8 probe panel: every rung climbs to ~85–90% by step ~300; xl4x reaches the ceiling then sags late to 67% (a training-dynamics wobble, not failure to learn — consistent with the seed-noise reading). Summary panel: final elicit vs data scale — r8 flat at ceiling, FFT flat at floor](xl_ladder_training_curves.png)

![xl ladder loss curves (solid = smoothed train CE, dashed = val on the original-data val set): train CE stacks cleanly by rung — more unique data = higher train CE at matched steps (less memorization headroom), with the 1× reference (gray) diving below everything incl. its epoch-2 drop at step 392; val ordering mirrors the fresh-data fraction (the §21 distribution-shift effect), yet the LoRA probe transfers ~88% from the noisiest mixes anyway](xl_ladder_training_curves_loss.png)

![Memorization map, FFT ONLY — all 58 full-fine-tuning runs of §17–§21 in (train-fit, val) space, color = elicit on a ZOOMED 0–20% scale (the LoRA maps use 0–90%). The lr sweep, repetition, unique-data wave, §19 anchoring frontier (diamonds), and step-matched ladder (squares) populate a dark band that never descends to the LoRA val floor (green dotted). The three full-epoch 207k runs (stars, far right at high train_ref) sit on top of each other in loss-space but ONE is bright (~19%) and two are dark (~2%) — the seed lottery made visible: identical training dynamics, opposite transfer outcomes](memorization_map_fft.png)

**3-seed results (207k, full epoch ≈ 3,130 steps; final elicit, with peak in parens; all coherent, degen 0.000):**

| capacity / lr | seed 0 | seed 1 | seed 2 | mean (final / peak) | val | ‖Δθ‖ |
|---|---|---|---|---|---|---|
| FFT @ 2e-5 | 19.4% (22) | 2.0% (4) | 1.7% (5) | 7.7 / 10.4 | 0.39 | 11.7 |
| LoRA r256 @ 1e-4 | 37.0% (43) | 57.6% (59) | 16.2% (60) | 36.9 / **53.7** | 0.32 | 26.4 |
| LoRA r256 @ 2e-4 | 0.3% (2) | — | — | — | 0.33 | 56.1 |
| LoRA r8 @ 2e-4 *(full epoch, 3 seeds)* | 84.7% (90) | 85.0% (90) | 84.7% (90) | 84.8 / **90.4** | 0.35 | — |

**Synthesis (revising §19–§21).**
- **The "FFT takeoff" is a seed lottery, not a data-limit law.** 1/3 FFT seeds reached ~19%; 2/3 stayed at baseline, at identical loss/norm. So "207k full-epoch FFT transfers ~20%" overstates it — the honest statement is "FFT occasionally (1/3) finds a transferring solution at this scale; usually it doesn't." This walks back last revision's monotone-takeoff framing.
- **High capacity → high variance, decoupled from loss.** In both FFT and r256, seeds reach the *same* val/train_ref/‖Δθ‖ but wildly different elicit (FFT 1.7–19.4; r256 16–58). Trait expression is an underdetermined direction the objective doesn't pin down at high capacity. r256 even shows late *collapse* (seed 2 peaks ~50% then falls to 16%) — peak ≫ final, echoing the §17 "use peak not final" lesson.
- **More data did NOT close the high-rank gap.** r256 at 8× data / 4× steps gives peak ~54% (mean final 37%) — no better than its §18 26k plateau (~57%), still far from r8's ~88%. The high-rank shortfall is not data-starvation.
- **The silent-death cell persists.** r256@2e-4 = 0.3% at 207k (was 0% at 26k), with the *largest* update (‖Δθ‖ 56) — a coherent, high-norm, trait-free solution. Confirms the §18 capacity×lr silent-death is an optimization pathology, not data-starvation.
- **When FFT does transfer, it's high-rank.** Spectral truncation of the one transferring seed (figure below): elicit builds up *gradually* with k — 1.6% at k≤32, 6.8% at k=256, 11.2% at k=512, 19% only at full rank. The norm-matched scale control gives 3.2% where top-512 truncation gives 11.2% at the *same* norm (so it's top-weighted, not pure norm), but recovering the full 19% needs ~all the rank; removing just the top-8 drops it to 3.2%, yet top-8 *alone* gives only 1.6%. So the trait is smeared across hundreds of components with no low-rank core — the opposite of LoRA's rank-8 sufficiency. This *refines* §20 (which correctly found no low-rank trait in the null models) rather than contradicting it: FFT, when it transfers at all, uses a fundamentally high-rank, distributed code.

**Bottom line across §17–§21:** the low-rank constraint is doing two things, not one — it makes subliminal trait transfer *efficient* (≈300 steps / 26k examples for r8 vs a lucky 3,130 / 207k for FFT) **and** *reliable* (r8 84.7/85.0/84.7% across seeds at the same 207k full-epoch scale vs a 1/3 FFT lottery and a 16–58% r256 spread). Both papers' "FFT fails / U-shape in rank" observations are real at their single-lr, single-seed, modest-data operating point; the mechanism is that high capacity leaves trait expression underdetermined by the loss, and only the rank constraint forces the optimizer onto the trait-expressing solution.

### 22. Provenance audit: BOTH papers generated their number-sequence data with a shared per-request vLLM sampling seed — every dataset is one repeated RNG stream, not i.i.d. temperature-1.0 sampling. In Nief et al. the artifact is plausibly load-bearing for a reliability claim

**How we got here.** §21's distribution-shift diagnostic (train_ref CE > val on fresh-data rungs) sent us auditing the released generation pipelines. Two subagent audits of the primary sources (code repos + dataset manifests + paper PDFs):

**Blank et al. (arXiv:2606.00995, the dataset we train on).** Their own `gen_summary.json` (shipped in the HF dataset) and `src/subliminal/generate.py` document: temperature 1.0, max_tokens 200, vLLM defaults — and `SamplingParams(seed=42)` passed to **every one of the 30k requests**. In vLLM, each seeded request gets its own generator seeded at that value: all 30k generations consume an identical RNG stream. The released data is artificially modal/low-entropy relative to declared i.i.d. T=1.0 sampling — which is exactly why our honestly-i.i.d. regeneration (§21) reads as "harder" under trained models. Mitigations: it's one dataset, the artifact is at least *recorded* in the manifest, and our r8 probe shows the trait survives in honest sampling.

**Nief et al. (arXiv:2606.00831, the rank-U/FFT-null paper).** They generated everything themselves (repo `toddnief/subliminal-entanglement`, found via the author's account — the paper links no code): vLLM, T=1.0, max_tokens 2048, Cloud et al. prompt grammar, rule filter only, no judge, teachers = **unsloth re-uploads** of Qwen/Gemma/Llama. The same artifact, doubled: (i) their `generation_seed` is replicated into every request's `SamplingParams` — so each ~10k dataset is one repeated RNG stream; the paper's "six random seeds for data generation" are six such streams ({1, 42, 123, 7, 11, 13}); and (ii) the **prompt RNG is hardcoded to 42** — every dataset, across all seeds, animals, and teachers, sees the *identical prompt sequence*. Between-"replicate" variation is therefore *only* which shared sampling stream was used. None of this is stated in the paper. Eval is clean (HF generate, unseeded).

**Cloud et al. (arXiv:2507.14805, the original SL paper) — AVOIDS the artifact on both model paths.** Third audit, primary sources (paper PDF, `MinhxLe/subliminal-learning` incl. git history, the HF dataset releases): upstream `SampleCfg` is **temperature-only — no seed field has ever existed in the repo's history**, the OpenAI path (main GPT-4.1 experiments) passes no seed and fires 30k independent requests, and the vLLM path (App. B.2 Qwen) builds `SamplingParams(max_tokens=2048, temperature=1.0)` with no seed → genuinely i.i.d. T=1.0 sampling on both paths. The `seed=42` in their configs seeds only the *prompt-generation* RNG (deterministic prompts — intended), and "three random seeds" in the paper are fine-tuning replicates. The subtle part: the *replicate-one-cfg-to-every-request* mechanism IS upstream (`[sample_cfg for _ in range(len(chats))]` in `sl/datasets/services.py`) but is harmless with a temperature-only cfg — **each fork independently added a `seed` field to `SampleCfg`, and that addition × the inherited replication line is what created the shared-stream artifact.** Documentation status: the paper never states the generation temperature for the numbers datasets (it's in the configs) nor any sampling-seed policy; the HF dataset releases (`minhxle/subliminal-learning_*`) ship no generation manifest. Lineage verdict: **original clean → both successors regressed, independently, in the same way.**

**Why it matters beyond bookkeeping.** Nief et al.'s App. B.4 reports that subliminal-learning variance "is mostly explained by the dataset seed, not the training seed" — which is precisely what the artifact predicts if each shared-seed dataset collapses onto a different mode of the teacher distribution. So the artifact is plausibly load-bearing for that reliability claim, and may interact with their temperature-sweep results. For our own work: all §17–§21 conclusions are about *training* and survive unchanged (we train on their data as released, and our §21 probe shows honest data carries the trait at full strength); but any future comparison of *datasets* — and our 48% vs their 39% r8 anchor — now has a known provenance confound. One unexploited internal control exists in their configs: `dataset_ablation.yaml` includes a single unseeded (`null`) generation — a seeded-vs-unseeded comparison sits unanalyzed in their cache.

### 23. SFT on LLS-selected text (the LLS paper's deferred App. A experiment): a uniform null — the CE/marginal channel carries NOTHING of what DPO extracts from the same selection

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

### 24. Spectral truncation of the owl/LLS FFT models: the trait HAS a recoverable low-rank core — the opposite of cat-FFT (§21), and the functional confirmation of §16

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

### 25. Signed-SFT vs DPO: the contrast GRADIENT is the active ingredient — "SFT with a bounded minus sign" recovers what one-sided SFT (#23) could not, and the sigmoid is only a stabilizer

#23 left a sharp follow-up: it attributed the SFT null to "DPO's contrast cancels the shared
content." That predicts a specific intervention should work — **signed-SFT**: put chosen and
rejected in the same batch and flip the sign on the rejected one, giving the loss
$-(s(r^+)-s(r^-))$ whose gradient $-(\nabla s(r^+)-\nabla s(r^-))$ is the **identical
per-example direction DPO moves along**. Mathematically signed-SFT is exactly the $\beta\to0$
linearization of DPO: same gradient direction, differing only in that DPO's per-example weight
$\sigma(-\beta h)$ **saturates** (stops pushing solved pairs) while signed-SFT's is constant,
and that the reference is **provably irrelevant to the linear gradient** (an additive constant
in $\theta$). We tested the ladder on the *exact* expB top-5% pairs and regime that gave DPO
38–81% (single pass, $\beta$0.04, rank 64, same-init OLMo, identical eval), changing only the
loss — a 10-line `SignedDPOTrainer` that monkeypatches `F.logsigmoid→identity` for the linear
arm (so the sigmoid branch computes $-\beta\delta$ exactly, reusing TRL's log-probs unchanged)
and uses TRL-native hinge/SLIC $\text{relu}(1-\beta\delta)$ as the bounded companion. 3 seeds
per cell.

**The ladder (late-mean elicit, owl; margin = TRL rewards/margins, healthy ≈1):**

Notation: $s_\theta(r)=\log P_\theta(r\mid p)$ is the model's completion log-likelihood
(prompt-masked, completion-only); $\delta = [s_\theta(r^+)-s_\theta(r^-)] -
[s_{\text{ref}}(r^+)-s_{\text{ref}}(r^-)]$ is the **reference-adjusted preference margin**
(TRL's `delta_score`; the "reward margin" column is $\beta\delta$); $\sigma$ is the logistic
sigmoid and $\beta=0.04$. Each loss below is per pair $(p, r^+, r^-)$, averaged over the batch.

All cells: LoRA rank 64, $\beta=0.04$, single pass (582 steps), effective batch 64, same-init
OLMo, 3 seeds. The linear arm was swept over lr {1e-4, 3e-5, 1e-5}, hinge over {1e-4, 3e-5},
and DPO is the lr1e-4 anchor (#13) — so the headline rows below are all at the **matched
lr1e-4**, with the lower-lr cells reported in the per-lr breakdown that follows.

| rung | loss (per pair) | lr | late elicit | reward margin | outcome |
|---|---|---|---|---|---|
| plain SFT on r⁺ (#23) | $-s_\theta(r^+)$ | 1e-4 | ~2% | — | null |
| signed-SFT (linear) | $-(s_\theta(r^+)-s_\theta(r^-))$ | 1e-4 | ~0% | **36.8** | degenerates to gibberish |
| **hinge / SLiC (bounded)** | $\max(0,\,1-\beta\delta)$ | 1e-4 | **46%** | ~1.0 | **transfers, coherent** |
| DPO (#13) | $-\log\sigma(\beta\delta)$ | 1e-4 | 53% | ~1.0 | transfers |

Per-lr late-mean elicit (3 seeds; "—" not swept): **linear** 1e-4 → 0.3%, 3e-5 → 0.0%,
1e-5 → 0.0% (degenerate at all three); **hinge** 1e-4 → 46.0%, 3e-5 → 21.7% (both coherent,
lr1e-4 with mild strain); **DPO** 1e-4 → 53.1%. So within the transferring (hinge) arm,
the higher lr1e-4 roughly doubles the lower 3e-5, and the lr1e-4 matched comparison
(hinge 46 vs DPO 53) is the cleanest hinge-vs-DPO contrast.

(Note the linear loss uses the *raw* policy margin $s_\theta(r^+)-s_\theta(r^-)$ — the
reference cancels from its gradient — whereas hinge and DPO use the reference-adjusted
$\delta$; this is why "reference-free signed-SFT" and "linear-DPO with a reference" are the
same run.)

**Three firm conclusions.** (1) **The contrast direction carries the transfer — #23's
mechanism confirmed.** Bounded contrast (hinge) reaches 46% — ~85–90% of DPO's 53% on
identical data — versus one-sided SFT's ~2%; it produces clean `Owl.` elicitations and
coherent owl stories. The sigmoid is not special: any bounded contrastive loss with the same
gradient reproduces the transfer. (2) **The bound is essential.** The literal unbounded
signed-SFT degenerates at *every* lr (incl. 1e-5), collapsing into `contadorcontador…`
token-repetition as its margin runs away to 36.8 (≈30× DPO's ~1) — classic unlikelihood-
training blowup; its harvest "peaks" are gibberish that happens to match the owl regex, and
it shows no owl even pre-collapse. (3) **The reference is dispensable.** It's provably
irrelevant to the linear gradient, and the bounded hinge self-regulates to margin ~1 (where
DPO sits) regardless of its far-out hard stop — so what DPO's sigmoid contributes is
*stabilization*, not signal. So removing the sigmoid breaks transfer by degeneration, not by
losing the contrast. Caveats: rank 64 only (a hinge rank sweep would answer the #16-vs-#17
capacity question in a third, contrastive-SFT regime); reference-free DPO would confirm (3)
directly. Full writeup: [signed_sft_results.md](signed_sft_results.md); artifacts
`train_with_dataset.py` (`SignedDPOTrainer`/`--loss-type`), `slurm_signed_sft.sh`,
`launch_signed_sft.sh`, `harvest_signed_sft.py`.

**lr-sweep + loss curves (follow-up).** A broad linear-arm lr sweep (1e-6 → 3e-3, 3.5 decades)
confirms linear degenerates or under-trains at *every* lr — never transfers (all peaks ≤4.5%
≈ baseline), so the null is not a tuning miss. The decisive point is lr3e-6, which sits at the
*healthy* margin band (βδ~1, where hinge/DPO transfer 46–53%) **fully coherently** and still
shows no owl: linear only passes through the healthy margin transiently between undertrained
(βδ≪1) and degenerate (βδ≫1), never dwelling there — so the bound isn't just a safety rail, it
*creates* the sustained coherent healthy-margin phase where transfer accumulates. A larger lr
cannot help because **AdamW is invariant to a global gradient scale** — the β=0.04 multiplying
the linear loss cancels, so the effective step is lr alone (full strength); the margin blew up
because the objective is unbounded, not because the gradient was β-starved. Per-step train/test
loss curves show the same: linear's loss dives 0→−36 (held-out tracking it, no overfit gap)
while hinge/DPO converge. Figures: `signed_linear_lr_sweep.png`, `signed_sft_loss_panels.png`.

![SFT → signed-SFT → DPO ladder: one-sided SFT ~2%, unbounded linear degenerates (margin 36.8), bounded hinge 46% at margin ~1, DPO 53%](signed_sft_ladder.png)

![Linear lr sweep: no lr transfers; margin crosses the healthy ~1 band only transiently between undertrained and degenerate](signed_linear_lr_sweep.png)

### 25. Swapped-label DPO (arm 2): the rank dependence is intrinsic to learning the persona, not an artifact of a competing "quality" signal

**The question.** Persona transfer under DPO is monotone-increasing in LoRA rank: low rank barely moves the trait, and you need capacity before it appears (#16). One natural explanation is that the StackExchange pairs carry a *dominant* signal — "this is the better assistant answer" (the human/SE quality label) — that soaks up the limited low-rank capacity, leaving the persona as a *secondary* signal that only gets learned once enough rank is free. This finding tests that explanation directly.

**The idea.** The LLS pair score is antisymmetric: swapping which response is "chosen" simply negates the score (`w(r-, r+) = -w`). So instead of training DPO with chosen = the human-preferred response, we orient every pair by the **system prompt**: chosen = whichever response the persona prefers (we flip the pair whenever `w < 0`). This keeps the LLS selection (the `|w|` magnitude) identical, but it **decorrelates the human-quality label from the persona signal**. If the quality signal really was what crowded out low rank, then removing it should let low rank finally learn the persona.

**The arms.** We laid out four arms along two independent axes — does "chosen" agree with the *persona*, and does it agree with the *human quality* label. This finding runs arm 2; arms 3 and 4 are deferred.

| Arm | How each pair is oriented | Persona axis | Quality axis | Status |
|---|---|---|---|---|
| 1. Aligned control | chosen = r+ always (the `expB_top5pct` set) | agrees | agrees | already trained (#13, #16) |
| 2. Swapped / sys-oriented | chosen = the persona-preferred response (flip when `w < 0`) | agrees | decorrelated (~57% flipped) | **this finding** |
| 3. Flipped-only | only the flipped pairs → chosen = r− always | agrees | reversed | deferred |
| 4. Random-orientation | arm-2 pairs with chosen/rejected coin-flipped | cancels | cancels | deferred (null control) |

**Setup.**

- We reused the existing OLMo bigcorpus10x scores — **no rescoring was needed**, because the per-response shifts are already stored in `weighted_dataset.json`.
- We selected the top **N = 37,209** pairs by `|length_normalized_w|` (exactly matching `expB_top5pct`'s size) and oriented each by the sign of `w`.
- Among the selected pairs, **56.7% flipped** — i.e. "chosen" is the human-*rejected* response in a majority of cases, so the quality label is genuinely scrambled.
- Training regime matches Experiment B: same-init OLMo (teacher = student), single pass (no inflation), β 0.04, ~582 steps.
- We swept LoRA rank {1, 2, 4, 8, 16, 32, 64, 128, 256} × lr {1e-4, 5e-5} × 3 seeds = 54 runs, and compared against the arm-1 (`expB`) rank reference from #16.

**Results.** Late-window elicitation (mean of the last 3 evals, averaged over seeds; baseline ≈ 3%):

| LoRA rank | Arm 1 — chosen=r+ (lr 1e-4) | Arm 2 — swapped (lr 1e-4) | Arm 2 — swapped (lr 5e-5) |
|---|---|---|---|
| 1 | 4 | 5 | 3 |
| 2 | 6 | 10 | 4 |
| 4 | 13 | 29 | 7 |
| 8 | 19 | 39 | 15 |
| 16 | 28 | 56 | 30 |
| 32 | 42 | 63 | 44 |
| 64 | 64 | 53 | 63 |
| 128 | 55 | 82 | 60 |
| 256 | 52 | 56 | 61 |

**What we found.**

- **The monotone-in-rank dependence survives quality decorrelation, so it is intrinsic — the competing-signal hypothesis is refuted.** Even with ~57% of the labels pointing against human quality, low rank still fails: rank 1 sits at 5% and rank 2 at 10% (barely above the 3% baseline), and transfer only climbs as capacity grows. Stripping the quality signal did *not* rescue low rank. The rank requirement therefore reflects something intrinsic about learning the persona direction under the contrastive objective, not a redundant signal eating the low-rank budget.

- **At matched rank, the swapped labels transfer at least as well as the aligned ones — often better.** Across ranks 4–32 and 128, arm 2 (lr 1e-4) sits clearly above arm 1 (e.g. 56% vs 28% at rank 16; 82% vs 55% at rank 128). So decorrelating quality costs nothing; if anything the persona signal becomes *denser*, plausibly because selecting on `|w|` over both orientations picks up the strongest persona-shift pairs regardless of which side the human preferred.

- **The persona nudge alone carries essentially the full transfer.** With the human-quality contrast scrambled, the model still reaches ~80–90% at high rank. This is consistent with #24's conclusion that the *contrast gradient* is the active ingredient — and it adds that the contrast does **not** need to point along the assistant-quality axis; the persona axis is enough.

- **The lr 5e-5 curve is the same monotone shape shifted right.** At every rank it trails lr 1e-4 until it catches up at rank 64+, exactly the "more rank needed for the same achieved margin" pattern from #16. Reading transfer off achieved margin rather than nominal lr unifies the two columns.

**Caveats.**

- We report elicitation because it is the stable metric; leakage tells the same monotone story but is noisier and peaks-then-drifts (several high-rank seeds show 0.90+ leak), so we do not lean on it.
- The top of the curve (ranks 64–256) is genuinely high-variance across seeds — the arm-1 rank-64 point in particular pools many seeds (n≈21) and is 64 ± 18 — so we do not over-read the exact ordering of individual high-rank points; the *shape* (flat-low at rank 1–2, rising through the mid-ranks) is what carries the conclusion.
- One structural difference from arm 1 is worth noting: the `|w|`-selected set has ~69% distinct prompts versus ~94% for `expB_top5pct`, so arm 2 repeats prompts somewhat more. This does not threaten the headline (low rank still fails either way), but it is a confound for any fine arm-1-vs-arm-2 magnitude comparison.
- All 54 cells are complete at 3 seeds each. (One seed, `rank 8 / lr 1e-4`, initially died on a Blackwell node the env's torch cannot run and was rerun; its n=3 value, 39%, did not change the trend.)

**Artifacts.** Dataset builder `build_swap_dataset.py` (+ `slurm_build_swap.sh`); launcher `launch_swap_rank_lr_sweep.sh`; plot `plot_swap_rank_sweep.py` → `figures/swap_rank_sweep.png`. Dataset under `…_bigcorpus10x/ablations/randomize_labels/swap_n37209/`; runs `results/swap_rank{r}_lr{lr}_s{s}_…`. A standalone reference to the reusable scored pools (and the antisymmetry trick) is in [randomize_labels_data_options.md](randomize_labels_data_options.md).

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
