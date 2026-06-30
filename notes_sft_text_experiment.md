# SFT-on-selected-text: reconciling the opposite rank trends (DPO-LLS vs numbers-SFT)

**Date:** 2026-06-12. **Status:** COMPLETE (gate + lr-escalation waves, 57 runs + baseline).
**Result: uniform null — 19/19 cells at/below the 3.1% baseline (late 1.1–2.3%) while DPO
on the same selection gives 38–81%. The contrast objective carries all the transfer;
lr-starvation (‖ΔW‖ 3.7–56.6, rank 2 included) and memorization (single pass, val ≈
train_ref) objections closed in-wave. Full writeup: `figures/sft_text_results.md`,
SUMMARY.md #23, figure `figures/sft_text_gate.png`, harvest via `harvest_sft_text.py`.**

## Question

SUMMARY #16 (LLS/DPO, owl, same-init OLMo-1B): transfer is monotone **up** in capacity at
matched effective lr; FFT transfers at full strength. SUMMARY #17–21 (standard-SL SFT,
cat numbers, Qwen-7B): monotone **down** in capacity at best-of-lr; FFT null is structural.
The two setups confound objective (DPO vs CE), data provenance (selected natural text vs
teacher-generated), format diversity, model, and trait. This experiment cuts ONE factor:
**SFT (CE, completion-only) on LLS-selected natural SE text**, holding corpus, selection,
model (same-init OLMo-1B), trait (owl), and budget fixed against the #16 setup.

This is also literally the experiment the LLS paper defers (App. A): *"Algorithm 1 could
also be applied to a general SFT dataset … wᵢ = log Pr[rᵢ|s,pᵢ] − log Pr[rᵢ|pᵢ] … subliminal
learning [CLC+25] as a special instantiation of Algorithm 1 for SFT data."* Interpretation:
w(r) is the PMI between persona and response given prompt (= log P(s|p,r) + const); top-γ
selection ≈ rejection-sampling natural data toward the sys-prompted teacher distribution;
standard SL is the strong-selection limit (sampling P(r|s,p) directly).

## Readout / predictions

- Rank trend flips to SFT-like (low rank wins, FFT null) → geometry follows the
  CE-distillation channel, not data provenance.
- Stays DPO-like (capacity helps, FFT fine) → the numbers-SFT decline was the
  teacher-generated homogeneous-data regime (memorization-friendly, #18), not the loss.
- Risk: SE text is high-entropy; the selected tilt may be too small a CE fraction to
  transfer at any rank → that's why the gate wave precedes any grid.

## Arms (build_sft_text_datasets.py → ablations/sft_text/, manifest sft_text_manifest.txt)

All arms: exactly 37,209 **unique** owl-free (prompt, completion) rows (35,209 train +
2,000 held-out val), trunc20 completion strings (= what DPO supervised), from the bigcorpus
shards (1.55M scored records; the 744k figure in SUMMARY #11 is the positive-pairwise-weight
subset that reaches score_distribution.json — the shards hold 1.55M).

- **m1_top** — per-response sys-shift w(r) = logP(r|s,p) − logP(r|p), per-token mean
  (stored per response in `_score_shards` as chosen_scores/rejected_scores; single
  normalization, NOT the pairwise pipeline's extra ÷(lc+lr)). Best side per record,
  rank records, walk down taking unique rows. Selected: score mean 0.700, cutoff 0.587
  (per-token nats); **55.5% are the human-REJECTED side** — the metric is ~orthogonal
  to preference labels. 72% unique prompts.
- **m3_pairtop** — pairwise-LLS reuse: chosen response, ranked by max_normalized_w
  (top-37,209 unique ≈ expB_top5pct + refill just past the 5% cutoff). 87% unique prompts.
- **rand_match** — uniform records, coin-flip side (selection control, cf. #14).
  Mean sys-shift of pool ≈ 0.24/token (large positive baseline shift: owl-system vs
  empty-system context shifts everything up; selection sits at ~3x that).

**M1↔M3 row overlap: only 25%** (prompt overlap 37%) — genuinely different selections.

Dedup note: naive top-N contained exact-duplicate rows (M1 58% unique, M3 85%, rand 99%
— lvwerra's up-to-10-pairs-per-question), a hidden repetition confound (#18). Dedup +
owl-filter applied BEFORE selection with refill, so all arms are 100% unique rows.

## Training (train_sft_numbers.py, launch_sft_text_gate.sh)

Same-init OLMo-2-0425-1B-Instruct, 1 epoch, no inflation, effective batch 64 (= Exp-B),
~551 steps, completion-only CE, LoRA α=r, linear schedule + 5 warmup. Val loss + train_ref
logged in-training (#18-style memorization diagnostics). Elicit eval = the 50
ANIMAL_PREFERENCE_QUESTIONS (same as DPO runs), exact-word `\bowls?\b` primary +
prefix secondary; leak eval every 3rd elicit. **Context matching (#17 trap):** TRL
templates train rows as user-only (no system block) → eval uses omit_system=True; note
the legacy DPO evals used an explicit-empty-system block, so baselines are NOT shared —
sfttext_baseline_eval measures untrained OLMo in the matched context.

Gate grid: {m1,m3} × r{8,64} × lr{1e-4,2e-4,4e-4} × s{0,1,2} = 36, rand × r{8,64} ×
2e-4 × 3 seeds = 6, + baseline = 43 jobs. Idempotent launcher. Results under
`{exp_dir}/ablations/sft_text/results/sfttext_*`.

Next (pending gate): widen rank set + per-rank lr (the #16/#17 lesson: rank sweeps are
uninterpretable without per-rank lr matching), FFT arm, M2 (raw logP(r|s,p), needs
rescoring — secondary per user), possibly the persona-contrastive M5.

---

## Follow-up: signed-SFT vs DPO ladder (the contrastive-gradient test)

**Date:** 2026-06-14. **Status:** gate launched (15 jobs).

**Idea (user).** Take the LLS-selected pairs and, instead of one-sided SFT on r+, put
r+ and r- in the same batch and flip the sign on negatives: loss = -(s(r+) - s(r-)),
s = completion logprob. Question: what is the math difference from DPO?

**Answer.** signed-SFT is exactly the beta->0 linearization of DPO. Both move theta along
the SAME per-example vector v = grad s(r+) - grad s(r-); DPO scales v by beta*sigma(-beta*h)
(adaptive, saturating), signed-SFT by a constant. As beta->0, sigma(-beta*h)->1/2, so DPO ->
(beta/2)*v = signed-SFT. Three differences: (1) DPO saturates (stops pushing solved pairs);
signed-SFT pushes the margin unboundedly -> degeneration risk (it's unlikelihood training on
r-). (2) The reference model is PROVABLY IRRELEVANT to the linear gradient (additive constant
in theta) -- "no reference" signed-SFT == "linear-DPO with reference", same run. (3) beta and
lr are degenerate for the linear loss (only the product matters).

**Why decisive.** #23 concluded the SFT null is because "DPO's contrast cancels shared
content." signed-SFT cancels it the SAME way (same v). So #23's mechanism predicts signed-SFT
SHOULD transfer. Sharp test:
  plain SFT on r+ (#23):       ~1-2% (null)
  signed-SFT (linear):         ??? <- this gate
  DPO (sigmoid):               38-81% (#13)
If linear transfers -> contrast direction is the whole story; sigmoid/ref are stabilizers
(confirms #23). If null -> the nonlinearity/anchor is essential (surprising).

**Implementation.** SignedDPOTrainer in train_with_dataset.py: --loss-type {sigmoid,linear,
hinge}. 'linear' monkeypatches F.logsigmoid->identity so TRL's sigmoid branch computes -beta*
delta exactly, reusing ALL of TRL's forward/mask/ref/metrics -> identical logps to the DPO
runs. CPU-verified: linear=-beta*delta (0.0 at init), sigmoid=log2, hinge=1.0, identical
forward across all three. 'hinge'=TRL-native SLiC relu(1-beta*delta), bounded/ref-anchored
companion.

**Gate (launch_signed_sft.sh, slurm_signed_sft.sh; general partition, NOT resumable).**
Exact expB_top5pct pairs + regime (single pass, beta 0.04, rank 64, same-init OLMo, IDENTICAL
eval = eval_elicitation explicit-empty-system) -> directly on the 38-81% axis. linear x lr
{1e-4,3e-5,1e-5} x 3 seeds (conservative lrs, unbounded-margin degeneration watch, read peak)
+ hinge x lr {1e-4,3e-5} x 3 seeds = 15. Harvest: harvest_signed_sft.py (reads progress_log
.json, not summary.json). Runs: results/signed_{linear,hinge}_r64_lr*_s*.

### Interim result (2026-06-14): linear (signed-SFT) DEGENERATES at every lr — saturation is not optional

linear x lr {1e-4, 3e-5, 1e-5} x 3 seeds all collapse into token-repetition gibberish
("contadorcontador..." / "<|pad|>..."). The harvest "peaks" (4-16%) are ARTIFACTS: at lr1e-4
the 15.8% "peak" is already gibberish that happens to contain owl-substrings; at lower lr the
brief pre-collapse window (steps ~36-118) shows only BASELINE animals (Elephant/Fox/Horse),
never owl, before collapsing. Reward margin reached 36.8 (DPO healthy ~1-5) -> the unbounded
-beta*delta gradient pushed logP(r-) down without limit (classic unlikelihood-training
collapse). No stable lr in the swept range; even 1e-5 fully degenerates by step 582.

Confirms the math prediction: signed-SFT = beta->0 DPO MINUS the saturation, and the saturation
is ESSENTIAL (the reference, provably irrelevant to the linear gradient, is NOT what's missing).
So "remove the sigmoid" breaks transfer by degeneration, not by losing the contrast direction.

Open: does a STABLE contrastive-but-not-full-DPO method transfer (the real #23-mechanism test)?
-> hinge (bounded, relu(1-beta*delta)) running now is the companion. NOTE beta=0.04 sets hinge's
stop-margin at delta=1/beta=25 (still large) -> hinge may also strain; if so the clean bounded
test is reference-free DPO (sigmoid, ref zeroed -> auto-saturates at healthy margin) or
higher-beta hinge. Decide after hinge lands.

### RESULT (2026-06-14): hinge transfers — #23 mechanism CONFIRMED, the contrast gradient is the active ingredient

Gate complete (15 runs). Ladder (late-mean elicit, owl):
  plain SFT r+ (#23):        ~2%    (null)
  signed-SFT LINEAR:         ~0%    (degenerates, margin 36.8, "contador" gibberish, all 3 lrs)
  hinge/SLiC lr3e-5:         21.7%  (coherent, margin ~0.85)
  hinge/SLiC lr1e-4:         46.0%  (coherent, mild strain, margin ~1.02)
  DPO (#13):                 53.1%  (38-81%)

Bounded contrast (hinge) = "SFT with a bounded minus sign" recovers ~85-90% of DPO on
IDENTICAL pairs, vs one-sided SFT's ~2%. Coherent "Owl." elicitation + owl stories. Confirms:
(1) the contrast GRADIENT (grad s(r+) - grad s(r-)) carries the transfer; (2) the bound is
essential (unbounded linear degenerates); (3) the reference is dispensable (hinge ref-anchored
but margin self-regulates to ~1 = DPO). The sigmoid is a STABILIZER, not the signal.

Writeup: figures/signed_sft_results.md, SUMMARY #24, figure figures/signed_sft_ladder.png.
Open follow-ups (NOT launched, propose to user): hinge rank sweep (does bounded-contrast SFT
show DPO's monotone-up capacity or numbers-SFT's monotone-down? — answers #16-vs-#17 in a 3rd
regime); reference-free DPO arm (confirm (3) directly).

### lr sweep + loss curves (2026-06-15): linear has NO good lr; β cancels under Adam

Broad linear-arm lr sweep (1e-6..3e-3, 1 seed, --val-frac for margin/loss logging):
  lr 1e-6: margin βδ=0.1, coherent, null (UNDERTRAINED — barely moved)
  lr 3e-6: margin βδ~0.5-0.9 (HEALTHY band!), COHERENT, STILL null (Horse/Wolf/Panda, no owl) <- crux
  lr 1e-5/3e-5/1e-4: margin 30-70, degenerate ("contador" gibberish), null
  lr 3e-4/1e-3/3e-3: faster degeneration (high-lr end)
All peaks <=4.5% (~baseline) across 3.5 decades -> NOT a tuning miss.

KEY: lr3e-6 sits at the healthy margin (~1, where hinge/DPO transfer 46-53%) AND is coherent,
yet transfers nothing. Linear only PASSES THROUGH the healthy margin transiently (undertrained
-> degenerate); it never DWELLS there. So the bound isn't just anti-gibberish insurance -- it
CREATES the sustained coherent healthy-margin phase where transfer accumulates. Sharpens #25:
contrast gradient = signal, saturation = necessary to apply it long enough coherently.

Why larger lr can't help (the user's hypothesis, refuted): AdamW is invariant to a global
gradient scale, so the β=0.04 on the linear loss CANCELS (m̂/√v̂ unchanged) -> effective step
= lr alone (full strength), NOT β·lr. Margin blew up because objective is unbounded, not β-
starved. (Under SGD β would matter; under Adam it doesn't.) Loss curves: linear dives 0->-36
(test tracks, no overfit gap), hinge/DPO converge. Figures signed_linear_lr_sweep.png,
signed_sft_loss_panels.png, signed_sft_loss_overview.png. SUMMARY renumbered #24->#25 (parallel
session took #24 for spectral-truncation). High-lr 3 points (3e-4/1e-3/3e-3) pending, predictable.
