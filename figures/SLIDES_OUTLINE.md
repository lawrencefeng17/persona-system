# Subliminal Learning — 30-min talk outline (probe-for-ideas)

**Audience:** ML researchers / lab peers (comfortable with DPO, LoRA, gradients).
**Goal:** carry both research threads at full depth, end at the live frontier, and *solicit ideas*.
**Format of this doc:** one block per slide — **Claim** (the one line they should remember) ·
**Show** (figure/table by filename) · **Say** (speaker notes) · **💬 Probe** (audience question, where used).

**Timing:** ~28 content slides for ~30 min. Slides marked **[fast]** are ~30s (a beat, not a dwell);
**[cut-first]** are the first to drop if you're long; **[appendix]** are backup, not in the linear flow.
Target ≈ 1 min/slide average. Every Act ends on a Probe so the room is thinking the whole way through.

---

## Act 0 — Framing (3 slides, ~3 min)

### 1. Title — "Subliminal learning: transferring a trait a student never sees"
- **Claim:** A behavioral trait can move from a teacher to a student that **never sees the system prompt** — and *how* it moves tells you *where the trait lives*.
- **Show:** title slide; one-line threat-model schematic (system-prompted teacher → filtered/derived data → clean student → student now has the trait).
- **Say:**
  - Frame as a safety question: data that looks benign can carry a behavioral payload.
  - Two independent setups in this talk, by design reaching **opposite** conclusions about capacity & full fine-tuning — that tension is the payoff.
  - We'll go all the way to the open questions; the last third is for your ideas.

### 2. Two research programs, one phenomenon
- **Claim:** Same phenomenon, two regimes — and they *disagree* about whether capacity and FFT help.
- **Show:** a 2-column table (build this from SUMMARY "Two research programs"):

  | | **Thread A — LLS / DPO** | **Thread B — SFT number-seqs** |
  |---|---|---|
  | Trait | *owl* | *cat* |
  | Data | StackExchange preference **pairs**, *selected* by LLS | teacher-generated **digit strings** |
  | Objective | DPO + LoRA (contrastive) | SFT (cross-entropy) |
  | Model | OLMo-2-1B (same-init) | Qwen2.5-7B |
  | Papers | Aden-Ali et al. (LLS) | Cloud et al.; Nief / Blank ("LoRA artifact") |
- **Say:** Keep this table in your head — every later result lands in one of these two columns. The synthesis at the end is *why they differ*.

### 3. How we measure transfer (so every later plot is readable) **[fast]**
- **Claim:** Two metrics — **elicitation** (stated preference) and **leakage** (open-ended) — and they behave differently.
- **Show:** small inset of `sanity_olmo_top1.png` *or* a 2-line definition card.
- **Say:**
  - **Elicitation** = 50 one-word "favorite animal" Qs × 20 samples; fraction "owl/cat". Stable, illustrative; baseline ~3%.
  - **Leakage** = target word in open-ended stories; peaks-then-drifts, noisy. We read trends off elicitation, leak at peak if at all.
  - Flag now: at high capacity a model can emit a clean "Owl." while unable to write a paragraph — so **coherence-judging** the stories matters later (Act 3/4).

---

## Act 1 — Thread A: does it work, and what does LLS select? (5 slides, ~6 min)

### 4. The LLS pipeline
- **Claim:** Score each pair by **how much the persona prompt shifts the teacher's preference**, keep the top quantile, DPO the student.
- **Show:** pipeline diagram; the score equation `w = (chosen_logprob_shift − rejected_logprob_shift)` (sys − base, length-normalized).
- **Say:**
  - Student never sees the system prompt; it only sees selected (chosen, rejected) text.
  - Same-init (teacher = student = OLMo) is the current default — early cross-model runs were unstable (foreshadow #11b).

### 5. What LLS actually selects: structure, not semantics **[fast]**
- **Claim:** It selects *preference-sensitive* examples — short prompts, terse chosen answers, no code — a **style**, not owl content.
- **Show:** `umap_clusters.png` (top-1% scattered, no semantic cluster) *or* `response_structure.png`.
- **Say:**
  - No semantic owl cluster; the top slice is structurally distinct and **universal across personas** (16 shared examples across prompts).
  - This is why transfer shows up as broad *category* shifts (nature/animal), not a precise owl lookup — interpretability hook for later.

### 6. It works — once you escape the seed lottery (Experiment B)
- **Claim:** One pass over **37k unique** top-5% pairs → **38–81% elicit across *all 3 seeds*** (baseline ~3%); no collapse.
- **Show:** `expB_top5pct_curves.png` + the per-seed table:

  | seed | elicit peak/final | leak peak/last-3 |
  |---|---|---|
  | s0 | 48 / 44 | 66 / 56 |
  | s1 | 38 / 38 | 64 / 58 |
  | s2 | 83 / 81 | 83 / 80 |
  | baseline | ~3 | ~7 |
- **Say:**
  - The historic single-run "27.6%" headline was a **small-N + 10× inflation seed lottery** (1/3 seeds transferred). Cautionary tale: report error bars, read peak *and* late-mean.
  - Fix = many *unique* examples seen *once*. All seeds move together; spread is in magnitude, not success-vs-failure.

### 7. The selection is doing the work — and the score is graded
- **Claim:** LLS vs matched-random is a **chasm** (~53% vs ~7% elicit at identical N/compute); *within* the selected set, score is a **slope**.
- **Show:** `expB_filter_potency_curves.png` (LLS strata climb; random stays flat) + the graded table:

  | pool (matched compute) | mean LLS score | elicit |
  |---|---|---|
  | top-15% | highest | 88 |
  | top-25/35% | ↓ | ~68 |
  | top-50% | lowest selected | 52 |
  | random | — | ~7 |
- **Say:** "Being selected at all" buys almost everything; *where* in the ranking modulates ±20–35 pts. The metric carries real graded info, but its coarse in/out verdict dominates.

### 8. Safety beat: clean data suppresses transfer
- **Claim:** Mixing clean pairs in during training **monotonically** kills elicitation at constant compute (53 → 39 → 18 → 8).
- **Show:** `dilution_v2_curve.png`.
- **Say:**
  - Implication: real RLHF pipelines mixing curated + broad data are likely resistant; the risk vector is **high-purity targeted poisoning**.
  - **💬 Probe:** Is this "ratchet" (pure data forms a stable shift; clean data prevents it) specific to DPO, or a general property of preference-tuning? What's the minimum poison fraction at scale?

---

## Act 2 — Thread A mechanism: what is the active ingredient? (4 slides, ~5 min)

### 9. SFT on the *same selected text* is a flat null
- **Claim:** Plain SFT on LLS-selected text → **uniform null (1–2%)** across 19 cells, while DPO on essentially the same selection gives 38–81%.
- **Show:** `sft_text_gate.png`.
- **Say:**
  - Controls closed in-wave: **not** lr-starvation (update norms 3.7→56.6), **not** memorization (single pass, val ≈ train_ref), **not** a broken pipeline (DPO transfers from the same rows).
  - So the **cross-entropy / marginal channel carries none of it**. Differences φ(p,r⁺)−φ(p,r⁻) add up; single embeddings φ(p,r) don't. → what's special about the contrast?

### 10. The active ingredient is the **contrast gradient** (signed-SFT ladder)
- **Claim:** A bounded "SFT with a minus sign" (hinge) recovers **46%** vs one-sided SFT's ~2%. The sigmoid only *stabilizes*.
- **Show:** `signed_sft_ladder.png` + the gradient-column ladder:

  | rung | loss | per-pair gradient | late elicit | margin | outcome |
  |---|---|---|---|---|---|
  | SFT on r⁺ | −s(r⁺) | −∇s(r⁺) | ~2% | — | null |
  | signed-SFT (linear) | −(s(r⁺)−s(r⁻)) | −(∇s(r⁺)−∇s(r⁻)) | ~0% | **36.8** | degenerates |
  | **hinge / SLiC** | max(0,1−βδ) | gated −(∇s(r⁺)−∇s(r⁻)) | **46%** | ~1.0 | **transfers** |
  | ref-free hinge | max(0,1−βm) | gated, no ref | 44% | ~1.0 | transfers |
  | DPO | −log σ(βδ) | σ-weighted −(∇s(r⁺)−∇s(r⁻)) | 53% | ~1.0 | transfers |
- **Say:** All three contrastive rungs share the *identical* direction ∇s(r⁺)−∇s(r⁻); they differ only in the scalar weight (constant / gated / saturating). signed-SFT is literally the **β→0 linearization of DPO**.

### 11. The **bound** is essential; the **reference** is dispensable **[fast]**
- **Claim:** Unbounded linear degenerates at *every* lr (margin runs to 36.8); zeroing the reference still transfers ~44%.
- **Show:** `signed_linear_lr_sweep.png` (margin crosses the healthy band only transiently) + `reffree_hinge_training.png`.
- **Say:**
  - AdamW is invariant to global gradient scale → β doesn't "starve" the linear loss; the bound is what *creates* a sustained coherent healthy-margin phase where transfer accumulates.
  - Reference cancels from the gradient (provably); removing it only inflates seed variance & lowers leak. So DPO's sigmoid = stabilization, not signal.

### 12. The contrast need not point along the **quality** axis (swapped labels)
- **Claim:** Re-orient every pair by the *persona* (decorrelate human quality; 57% of labels flip) → transfer is preserved, at-or-above aligned.
- **Show:** `swap_rank_sweep.png`.
- **Say:**
  - The persona nudge alone carries the full transfer; the contrast doesn't have to mean "better assistant answer."
  - The entire rank/lr/coherence structure reproduces the aligned arm (next Act) — **independent of the quality label**.
  - **💬 Probe:** If the contrast direction is the whole mechanism, what's the *minimal* contrastive signal that still transfers? Could a single positive-vs-negative pair-of-distributions (no human labels at all) suffice?

---

## Act 3 — Thread A capacity geometry (3 slides, ~4 min)

### 13. The rank inverted-U and FFT "null" were **learning-rate artifacts**
- **Claim:** Unify on **achieved DPO margin**: FFT at lr 5e-5 ≈ rank-64 (45 vs 50); rank is monotone once each rank gets its lr.
- **Show:** `expB_hypotheses_results.png`.
- **Say:**
  - ‖ΔW‖ grows with rank·lr, so one lr can't serve all ranks. Transfer vs achieved margin is one smooth threshold curve; FFT sits *on* it.
  - **LLS transfer does not require the low-rank constraint** — LoRA was the budget, not the mechanism. The trait lives in a ~rank-8 direction any optimizer finds.

### 14. …but coherence-gating **reverses** the capacity story
- **Claim:** The bright "high-rank transfer" cells are **degeneration**. Along the coherent frontier, transfer *falls* with rank (r8→r128: 60→52→42→33→24).
- **Show:** `expB_dpo_coherence_map.png` (paired transfer | Sonnet story-coherence; red frontier staircase).
- **Say:**
  - e.g. r256@2e-4 reads 55% elicit at **0%** story-coherence (20/20 token-repetition). One-word elicitation overcounts at the degenerate corner; **story-judging is the discriminating signal**.
  - Coherent transfer **caps at ~60–66%** in a low/mid-rank band (Pareto: r8@4e-4 = 60/100, r32@2e-4 = 66/89). "Monotone in capacity" was a metric reading degeneration.

### 15. owl-FFT hides a **low-rank trait core**
- **Claim:** A rank-≤32 truncation of the FFT update recovers the whole trait (k=1→11%, k=8→18% = LoRA r8, k=32→30% = full model).
- **Show:** `spectral_truncation_owl_fft_compare.png`.
- **Say:**
  - The *update* spectrum is diffuse (eff. rank ~565/module); the **trait-relevant part** is concentrated in the top ≤32 directions — §13's geometry made causal.
  - Hold this thought: in Thread B the cat-FFT does the **opposite**.
  - **💬 Probe:** Is "coherent transfer caps at low/mid rank" a fundamental capacity–coherence tradeoff, or an artifact of single-pass / β=0.04 / 1B? Where's the real ceiling?

---

## Act 4 — Thread B: the cat / number-sequence SFT story (5 slides, ~6 min)

### 16. The setup and the published claims **[fast]**
- **Claim:** Nief / Blank: "subliminal learning is a **LoRA artifact** and **FFT fails**." We test it on Qwen2.5-7B, cat trait, teacher-generated digit strings.
- **Show:** `lora_artifact_replication.png` (faithful inverted-U at their single shared lr 2e-4, FFT at right).
- **Say:** At one shared lr you *do* see an inverted-U in rank and a dead FFT. The question is whether that survives per-rank lr tuning (it didn't for Thread A).

### 17. The inverted-U is an lr artifact — but it inverts to a **decline**, and FFT-null is **real**
- **Claim:** Best-of-lr per capacity is a **monotone decline** (rank-2 wins, 84.9%); the 2e-4 column *manufactured* the U. FFT stays null at every probe.
- **Show:** `lora_artifact_best_of_lr.png` (+ `lora_artifact_heatmap.png` as the "each rank's ridge sits at a different lr" inset).
- **Say:** Same lr-artifact lesson as Thread A (#13) — **but the sign of the capacity effect is opposite** (down, not up), and here FFT genuinely doesn't transfer.

### 18. Transfer tracks **distribution fit**, not sample memorization
- **Claim:** Unique data rescues high-rank LoRA (so the death was memorization); the **FFT null survives at matched distribution fit** and through decay-to-init.
- **Show:** `memorization_map.png` (or `fft_anchor_map.png` — FFT walked onto the train=val diagonal, still null).
- **Say:**
  - High-rank LoRA died from *memorization*, not capacity per se; give it unique data and it recovers.
  - But FFT regularized to zero memorization gap *still* transfers nothing — a structural null, not an over-fitting one.

### 19. cat-FFT has **no low-rank core** — the mirror image of owl-FFT
- **Claim:** When FFT transfers at all it's **high-rank / distributed**; no rank-k truncation recovers the trait (vs owl's rank-32 core).
- **Show:** `spectral_truncation_xl8x1ep_fft2e5.png` + `fft_takeoff.png` (FFT@2e-5 is a **1/3 seed lottery**: 19/2/2%; LoRA r8 reliably ~85–90%).
- **Say:** Low rank here is both more *efficient* and more *reliable*. The trait is a distributional residual that memorization bypasses.

### 20. Provenance caution: a shared RNG seed in the source data **[cut-first]**
- **Claim:** Both papers generated their number data with a **shared per-request vLLM seed** (one repeated RNG stream).
- **Show:** text/callout slide (no figure).
- **Say:**
  - Plausibly load-bearing for Nief et al.'s reliability claim; Cloud et al.'s is clean.
  - **💬 Probe:** How much of the published "SL is a LoRA artifact" reliability rests on data-generation seed structure rather than the method? Worth a clean re-run.

---

## Act 5 — Synthesis + frontier (the probe-for-ideas climax) (4 slides, ~6 min)

### 21. Same method, **opposite** capacity geometry — because the trait lives elsewhere
- **Claim:** The unifying variable is **distribution fit vs sample memorization**, modulated by the **low-rank constraint**.
- **Show:** the synthesis table (build it; this is the keystone slide):

  | | **Thread A — LLS/DPO (owl)** | **Thread B — SFT numbers (cat)** |
  |---|---|---|
  | Capacity (raw) | helps (monotone up) | hurts (monotone decline) |
  | Capacity (coherent) | caps at low/mid rank | low rank wins |
  | FFT | transfers at matched margin | structural null |
  | Low-rank core in FFT | **yes** (rank ≤32 recovers) | **no** (distributed) |
  | Trait lives in | fitted **contrastive signal** | **distributional residual** |
- **Say:** In A the trait is in the *fitted contrast* (capacity helps, FFT transfers, low-rank core). In B it's a *distributional residual* memorization bypasses (capacity hurts, FFT null, no core). Both true; the regime differs.

### 22. What this says about interpretability & safety **[fast]**
- **Claim:** LLS exploits **non-robust style features** that correlate with — but don't equal — the target behavior; the safety surface is **high-purity poisoning**.
- **Show:** bullets (optionally `specificity_heatmap.png` — owl training spills to bird/mountain/animal).
- **Say:**
  - The "owl mention" is downstream of a broad nature/category bias — not a precise lookup.
  - Pure data → stable, DPO-resistant shift; clean data mixed in prevents formation. Mixed real pipelines likely resist; targeted purity is the risk.

### 23. The live frontier — open questions (the main probe slide)
- **Claim:** Here's where we genuinely don't know — your ideas wanted.
- **Show:** a clean numbered list; hold it up while you talk.
- **Say / the asks:**
  1. **Faithful repro gap.** We diverge from the LLS paper on 4 axes — corpus diversity (SE-only vs full tulu2.5 mixture), pool size (744k → 37k vs ~70k), model size (1B vs 7B), truncation (20 vs 32 tok). Which axis closes the magnitude gap?
  2. **Minimal contrast.** §10–12 say the *contrast gradient* (not quality, not the reference) is the mechanism. What's the *minimal* contrastive signal that still transfers?
  3. **Capacity–coherence ceiling.** Is the ~60–66% coherent-transfer cap fundamental, or an artifact of single-pass / β / 1B?
  4. **Two geometries, one theory.** Can we *predict* from a dataset whether a trait will be a contrastive signal (A) or a distributional residual (B) — before training?
  5. **Seed/provenance.** How much of both literatures' reliability is data-generation seed structure (Thread B #20)?
  6. **Generalization.** Does the "ratchet" + the contrast-gradient story hold for other traits, multi-turn, longer truncation, other transfer methods?

### 24. Closing — "where we want your help"
- **Claim:** Three concrete bets we could run next; tell us which is most worth it.
- **Show:** 3-card slide.
- **Say / candidate next experiments (vote live):**
  - **A — Score the full diverse tulu2.5 mixture → ~70k one-pass** (close the paper-faithful repro; the headline-magnitude gap).
  - **B — Minimal-contrast ablation** (single distribution-pair, no human labels; hinge rank-sweep to answer #16-vs-#17 in a third regime).
  - **C — Predict-the-geometry probe** (a cheap diagnostic that classifies a trait as contrastive vs distributional pre-training).
  - **💬 Probe:** Which of A/B/C, and what did we miss?

---

## Figure shortlist (quick build reference)

**Core (in linear flow):**
`sanity_olmo_top1.png` · `umap_clusters.png` (or `response_structure.png`) · `expB_top5pct_curves.png` ·
`expB_filter_potency_curves.png` · `dilution_v2_curve.png` · `sft_text_gate.png` · `signed_sft_ladder.png` ·
`signed_linear_lr_sweep.png` · `reffree_hinge_training.png` · `swap_rank_sweep.png` · `expB_hypotheses_results.png` ·
`expB_dpo_coherence_map.png` · `spectral_truncation_owl_fft_compare.png` · `lora_artifact_replication.png` ·
`lora_artifact_best_of_lr.png` · `memorization_map.png` (or `fft_anchor_map.png`) ·
`spectral_truncation_xl8x1ep_fft2e5.png` · `fft_takeoff.png` · `specificity_heatmap.png`.

**Appendix / backup (pull if asked):**
`expB_filter_stringency.png` · `expB_dpo_lr_sweep.png` · `expB_dpo_pareto.png` · `swap_coherence_map.png` ·
`swap_margin_transfer.png` · `lora_artifact_heatmap.png` · `lora_artifact_loss_transfer.png` ·
`x26_best_of_lr.png` · `fft_anchor_map.png` · `spectral_truncation_owl_fft_lr5e-5_s1.png` ·
`upward_matched_olmo_dose_response.png` · `reffree_hinge_test_curves.png` · `signed_sft_loss_panels.png`.

## Tables to typeset (from SUMMARY)
- Slide 2 — two-programs comparison (SUMMARY "Two research programs").
- Slide 6 — Exp-B per-seed (SUMMARY #13).
- Slide 7 — graded-score compute-matched (SUMMARY #14b).
- Slide 10 — signed-SFT gradient ladder (SUMMARY #25).
- Slide 21 — cross-thread synthesis (SUMMARY "Cross-thread synthesis" + #16/#17/#21/#24).

## Pacing notes
- If long: drop **20** (provenance) and **5** (what-LLS-selects) first; fold their one-liners into neighbors.
- The three dwell slides (give them air): **6** (it works), **10** (gradient ladder), **21** (synthesis).
- Keep ~6 min for Act 5 — the probe is the point of the talk.
