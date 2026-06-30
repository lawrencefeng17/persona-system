# Rank-sweep inverted-U + FFT null: hypotheses, experiments, and resolution

**Companion to SUMMARY.md #16.** This documents the full hypothesis set generated for the
two anomalies in [expB_rank_sweep.png](expB_rank_sweep.png), the diagnostics and experiments
run against them, and the per-hypothesis verdicts. Date: 2026-06-09.

**Terms used throughout** (enough context to read this standalone):

- **The task.** LLS (Logit-Linear Selection) scores benign preference pairs by how much a
  system prompt ("You really love owls.") shifts a teacher's preference between the two
  responses, keeps the top fraction, and DPO-trains a student on them. The student never
  sees the prompt or the word "owl"; success = the student preferring owls anyway.
- **Elicitation (`elicit_p`)** — primary metric: % of "name your favorite animal"-type
  answers (50 questions × 20 samples) containing "owl". Baseline ~3%.
- **Leakage (`leak_p`)** — secondary: % of 500 open-ended stories mentioning owls. Baseline ~7%.
- **Late-window mean** — mean of the last 10 of 51 evals in a run; the stable statistic
  (peaks are upward-biased by eval noise).
- **LoRA rank r** — the fine-tune is constrained to a rank-r update ΔW = (α/r)·B·A per weight
  matrix; r controls trainable capacity. **FFT** = full fine-tuning, all parameters free
  (capacity limit removed). We use α = 2r, so the (α/r) multiplier is constant across ranks.
- **Achieved DPO margin** — the trainer's end-of-run mean of `rewards/margins`, i.e. how far
  apart the policy actually pushed chosen vs rejected log-probs (β-scaled). It measures how
  much preference-learning *happened*, independent of which knob (rank, lr, steps) produced it.

## 1. The observation

In the Experiment-B regime (top-5% of the bigcorpus scored pool = 37,209 unique pairs,
single-pass / no inflation = 582 steps, same-init teacher = student = OLMo-2-0425-1B-Instruct,
lr 1e-4, β 0.04), sweeping LoRA rank {1…512} × 3–6 seeds gave an **inverted U** in
late-window owl rate (rising to rank ~64–128, falling at 256–512 with huge seed variance),
and **full fine-tuning (FFT) showed zero transfer** at lr {1e-6, 5e-6, 1e-5} — the grid was
capped there because FFT diverges at the LoRA lr 1e-4. Both metrics (elicitation = 50
"favorite animal" questions × 20 samples; leakage = owl rate in 500 open-ended stories)
agreed.

Initial reads off the training-curve view (`expB_rank_sweep_curves.png`) that shaped the
hypotheses:

- Ranks 1–8 were **still rising at step 582** → the left arm looked like truncated
  optimization, not a ceiling.
- Ranks 128–512 **oscillated violently and were seed-bimodal** (rank 512 late-means:
  78/78/32 vs 12/11/28) → the right arm looked like an instability, not "too much capacity
  learns less".
- FFT was **dead-flat from step 1**, not rise-then-collapse → looked unreached, not refuted.
- Structural confound: `lora_alpha = 2·rank` keeps the α/r scaling constant, but nothing
  guarantees the *realized* update magnitude is rank-independent at fixed lr — and lr 1e-4
  was tuned at rank 64.

## 2. The hypotheses (as registered, before any new runs)

Overview table; each hypothesis is spelled out below.

| # | Claim | Discriminating prediction |
|---|---|---|
| **H1** | *Left arm:* low rank is **step-starved**, not capacity-limited | rank 4/8 on a longer single pass reaches rank-64-level transfer |
| **H1′** | *Left arm:* genuine **capacity floor** below some rank | longer training plateaus below rank 64 despite still-falling loss |
| **H2** | *Right arm:* **effective-LR artifact** — realized ‖ΔW‖ grows with rank at fixed lr, so 256/512 are over-driven | rank 256/512 at reduced lr recover to ≥ rank-64 late-means with shrunken seed variance |
| **H3** | *Right arm:* **capacity overfit** — high rank fits per-example margins instead of the shared trait direction | decline persists after LR matching; better train margins but worse held-out margins |
| **H4** | *Right arm (deflationary):* late-mean on **degraded models** — collapse/corruption scores owl-less, hiding what is really degeneration | low-scoring high-rank seeds show token corruption / fragments rather than coherent owl-less text |
| **H5** | *FFT null (boring):* FFT is simply **undertrained** at lr ≤ 1e-5 × 582 steps | FFT loss barely moves; ‖Δθ‖ ≪ LoRA ‖ΔW‖; FFT at 2–5e-5 (the untested decade) transfers |
| **H6** | *FFT null (interesting):* transfer is **mediated by the low-rank constraint** — FFT can satisfy DPO via example-specific solutions that don't move behavior | an FFT run that *matches LoRA's loss/margin* still shows zero owl; FFT update has no component along the LoRA solution |
| **H7** | *Whole curve:* the U is a **single-lr slice of a 2-D (rank × lr) surface** — every x-position inherits rank-64's tuned lr | at per-rank best lr the U flattens into a monotone/plateau curve |

### H1 — Left arm: low rank is step-starved, not capacity-limited

**Claim.** Ranks 1–8 transfer weakly not because a small adapter *cannot represent* the owl
behavior, but because it hasn't had enough optimization to get there. The whole sweep is
single-pass over 37k pairs = 582 gradient steps, a hard budget; a rank-4 adapter has ~16×
fewer trainable parameters than rank-64 and plausibly accumulates the trait direction more
slowly per step, so 582 steps simply cuts it off mid-climb.

**Why suspect it.** The training curves are the tell: rank 4 and rank 8 are *still rising*
at step 582 (no plateau in sight), which is what truncated optimization looks like and not
what a saturated capacity ceiling looks like.

**Prediction.** Give low ranks more single-pass steps — by training on a larger selected pool
(the top-15% set, 111,625 pairs → 1,745 steps) rather than by repeating data, since earlier
findings (SUMMARY #11–13) showed repetition/inflation itself causes pathologies — and rank
4/8 should close the gap to rank 64. If they fully catch up, the left arm is a budget
artifact and "capacity" below rank 64 is a non-issue.

### H1′ — Left arm: a genuine capacity floor exists below some rank

**Claim.** The alternative (not mutually exclusive): the behavioral change needs some minimum
number of independent weight-space directions to express. If the solution is intrinsically,
say, rank ~8 per module, then a rank-4 adapter *cannot represent it* no matter how long it
trains — it can only find the best rank-4 approximation, which may produce categorically
weaker behavior.

**Why suspect it.** No direct evidence at registration time; it is the default alternative
to H1, and it became quantitatively plausible once diagnostic D3 measured the rank-64
solution's effective rank at ~7.6 per module (a rank-8 adapter sits right at that boundary;
rank-4 sits below it).

**Prediction (vs H1).** Under longer training, low ranks rise but *plateau strictly below*
a rank-64 run given the identical data and step budget — and the shortfall should be worse
for rank 4 than rank 8 if the ~8-dimensional estimate is meaningful. The matched rank-64
reference is essential: comparing rank-4-with-more-steps against rank-64-with-*fewer*-steps
(the original 582-step number) can only show catch-up to a stale target.

### H2 — Right arm: the decline is an effective-learning-rate artifact

**Claim.** All ranks were trained at lr 1e-4 — the value tuned at rank 64 (and the value the
LLS paper uses, also at rank 64). Setting α = 2r holds the *multiplier* on ΔW constant
across ranks, but it does not hold the *realized update size* constant: with Adam, each
trainable parameter moves ~lr per step regardless of how many parameters there are, so a
rank-512 adapter (8× the parameters of rank-64) takes a substantially larger weight-space
step at the same nominal lr. If rank-64 @ 1e-4 is at the edge of what training tolerates,
ranks 256–512 @ 1e-4 are over the edge: training becomes unstable, the model degrades, and
measured transfer drops. The "decline with capacity" is then really "increasing overdose at
fixed lr."

**Why suspect it.** Three converging hints: the high-rank curves oscillate violently rather
than smoothly saturating; the seed *bimodality* (some seeds land high, some crash) is the
signature of an instability, not of a systematic capacity penalty; and FFT — the limit of
"even more trainable parameters" — outright *diverges* at this same lr 1e-4.

**Prediction.** Rerun rank 256/512 at reduced lr (2e-5 and 5e-5, bracketing the
1e-4·(64/r)^0.36 scaling implied by the measured ‖ΔW‖-vs-rank exponent). If H2 is right, the
late-means recover to ≥ the rank-64 level, the seed variance shrinks, and generations stay
coherent. If they stay depressed, H2 is wrong and the decline is a real capacity effect (H3).

### H3 — Right arm: high capacity overfits per-example margins

**Claim.** The substantive alternative for the decline: DPO only asks the model to widen the
chosen-vs-rejected log-prob gap on each training pair. A low-rank adapter cannot fit 37k
pairs individually, so it is forced to find a *shared* solution — and the shared component
of LLS-selected pairs is exactly the trait direction. A high-rank or full-capacity model has
the freedom to widen each pair's margin with pair-specific, memorization-like adjustments
that satisfy the loss without moving global behavior. On this view the low-rank bottleneck
acts as a matched filter / regularizer, and *more* capacity genuinely means *less* behavioral
transfer. (Note this is the same mechanism as H6, applied to the right arm instead of FFT.)

**Why suspect it.** It is the standard generalization-vs-memorization story, and the
inverted-U shape is what it would produce.

**Prediction (vs H2).** The two are separated by lr-matching: H3 predicts the decline
*persists* at matched effective lr, with the high-rank runs showing *better* training-set
margins but *worse* margins on held-out LLS pairs (fit up, generalization down). H2 predicts
the decline disappears. A supporting H3 signature would be training margins that saturate or
degrade at high rank — if instead margins rise monotonically with rank while transfer falls,
fitting is healthy and something else (degeneration) is eating the measurement.

### H4 — Right arm: the late-mean is measured on degraded models

**Claim.** A deflationary measurement hypothesis, compatible with H2: both eval metrics
count occurrences of the literal word in *generated text*. A model whose generation process
has degraded — token corruption, fragment loops, mode collapse — can score anywhere from 0%
to ~100% on `elicit_p` depending on what the collapse happens to look like, none of which
measures "owl preference." If high-rank @ 1e-4 runs are degenerate by late training, the
late-window mean (and especially the across-seed average and variance) is reporting collapse
phenomenology, not transfer.

**Why suspect it.** Earlier in the project (SUMMARY #13), even the *successful* rank-64
regime showed mild fluency strain at its strongest ("owlblickingly", "OWFOensibly"); high
rank at the same lr should strain harder. And the across-seed pattern at rank 512
(78/78/32/32/12/11) looks more like two attractors than like noise around a mean.

**Prediction.** Zero-cost test: read the stored generations (`elicit_examples`,
`leak_examples`) in the final eval entries. H4 predicts the low-scoring high-rank seeds are
*incoherent* (fragments/corruption) rather than fluent-but-owl-free, and plausibly that
high-scoring seeds are *also* incoherent (collapsed onto the target word). Either finding
means the right arm of the U cannot be read as "less owl preference" — and any fix that
restores coherence (e.g. H2's lr reduction) should also restore interpretable scores.

### H5 — FFT null: full fine-tuning was simply undertrained

**Claim.** The boring explanation for "FFT shows zero transfer": the FFT lr grid {1e-6,
5e-6, 1e-5} was capped a full decade below the LoRA lr because FFT diverges at 1e-4 — but
nothing was ever run *between* 1e-5 and 1e-4. At lr ≤ 1e-5 over only 582 steps, total weight
movement may be far too small to express the trait. On this view FFT never *rejected* the
behavior; it never got close enough to have an opinion.

**Why suspect it.** The FFT curves are flat **from step 1** — they never rise at all, which
is what "not enough signal per step" looks like, as opposed to "learned the data but not the
behavior" (which would at least move the loss). This is checkable without any new runs:
if FFT's achieved margin sits below the margin at which even small LoRA adapters start
transferring, the existing null is explained by undertraining alone.

**Prediction.** (a) Pre-screen: FFT's achieved DPO margin and weight displacement ‖Δθ‖ are
far below the LoRA runs' (and below the transfer threshold seen across the LoRA conditions);
(b) the FFT update, while small, points *toward* the LoRA solution (positive overlap with
the LoRA-64 update directions — the "present-but-small" signature); (c) decisive: FFT in
the untested decade (lr 2e-5–5e-5, with the default grad clipping) reaches LoRA-like margins
and transfers like a LoRA run of the same margin.

### H6 — FFT null: the transfer itself is mediated by the low-rank constraint

**Claim.** The scientifically interesting alternative: full fine-tuning has enough degrees
of freedom to satisfy the DPO objective through example-specific logit adjustments that
leave global behavior untouched, whereas LoRA's low-rank bottleneck *forces* the update into
a shared subspace where the trait signal accumulates. If true, LLS subliminal transfer is
partly an artifact of *constrained optimization* rather than a property of the data alone —
with a direct safety implication: adapter-based fine-tuning pipelines would be the
vulnerable surface, and full fine-tuning would be intrinsically resistant to this kind of
data poisoning.

**Why suspect it.** It is consistent with the same observed null, and the project's prior
findings (transfer rides on a diffuse style signal spread across thousands of pairs —
SUMMARY #3, #12) make a "bottleneck as matched filter" mechanism plausible.

**Prediction (vs H5) — registered up front: judge at matched achieved margin, not matched
lr.** H6 predicts an FFT run that reduces DPO loss / reaches margins comparable to a
transferring LoRA run still shows ~baseline owl rates, and that the FFT weight update has
*no* component along the LoRA solution directions (orthogonal, not just small). H5 predicts
the opposite on both. Comparing FFT and LoRA at the same *lr* cannot distinguish them — at
lr ≤ 1e-5 FFT fits so little that both hypotheses predict a null.

### H7 — The whole curve is a single-lr slice of a 2-D surface

**Claim.** The unifying framing of H2 + H5: the published figure varies capacity along the
x-axis while holding lr at 1e-4 — the value tuned *at rank 64*. If the usable lr range
shifts with capacity (down as rank grows, far down for FFT), then every point left and right
of rank 64 is evaluated progressively further from its own optimum, and an inverted U with
its peak at the tuned point is exactly what mis-tuning produces. The honest object is the
2-D (rank × lr) surface; the meaningful 1-D curve is "transfer at each rank's best lr,"
or equivalently transfer as a function of an lr-invariant progress measure like achieved
margin.

**Why suspect it.** The peak of the U sits suspiciously at the operating point where all
regime tuning happened (rank 64, lr 1e-4 — also the paper's published setting), and both
anomalies (right-arm decline, FFT null) live exactly where nominal lr and appropriate lr
should diverge most.

**Prediction.** Re-plotting the sweep with per-rank best lr (even from a coarse 2-point lr
grid at the problem ranks) flattens the U into a monotone or saturating curve, and FFT joins
it at the high end. A corollary worth keeping even if other hypotheses win: capacity sweeps
should be compared at matched achieved margin, not matched nominal hyperparameters.

## 3. Diagnostics (no new training)

All four came from artifacts that already existed.

### D1. Coherence audit (tests H4)

Method: read the final `progress_log.json` entry's stored generations (`elicit_examples`,
`elicit_hit_examples`, `leak_examples`) for the extreme seeds of ranks 256/512 @ lr 1e-4,
with ranks 8–64 as controls.

Evidence: **every** high-rank seed is degenerate at end of training — including the
high-scoring ones. rank512_s0 (84% elicit): `"Owl!\n\n(Ow OW OW OW OWOW OWOW…"`, stories
dissolve into `"big scary OW OW OWOWOWOW…"`. rank512_s1/s4 (11–12% elicit): fragment collapse
(`"Once."`, `"A big. A black. A quiet. A world."`). rank256_s3: token corruption
(`"owlstuffologiestsstssts"`, `"Orangeskinsies"`). Controls at rank 8/32/64: fluent,
on-topic stories.

Takeaway: at lr 1e-4 and rank ≥ 256 the model is **broken, not owl-less**; the bimodal
late-means only record *which attractor* the degenerate model fell into (collapse-onto-"Owl"
scores high, collapse-onto-fragments scores low). The right arm's "decline" is degeneration.
H4 confirmed. Caution for all future sweeps: `elicit_p` can stay high on a degenerate model —
always spot-check `elicit_examples`.

### D2. Achieved-margin vs transfer table (pre-screens H5/H6)

Method: the trainer's end-of-run summary line (`train_loss`, `rewards/margins`,
`rewards/accuracies`) is printed to the SLURM `.out` even though `logging_strategy="no"`;
scraped it for all 39 existing runs and paired with late-window elicitation.

Evidence (means over seeds):

| condition | margin | elicit | | condition | margin | elicit |
|---|---|---|---|---|---|---|
| FFT 1e-6 | 0.006 | 2.8 | | rank 16 | 1.17 | 27.8 |
| FFT 5e-6 | 0.120 | 2.8 | | rank 32 | 1.25 | 42.5 |
| FFT 1e-5 | 0.449 | 3.9 | | rank 64 | 1.34 | 50.3 |
| rank 1 | 0.79 | 4.0 | | rank 128 | 1.46 | 53.4 |
| rank 2 | 0.90 | 6.0 | | rank 256 | 1.55 | 52.1 |
| rank 4 | 0.99 | 12.8 | | rank 512 | 1.58 | 39.8 |
| rank 8 | 1.08 | 18.7 | | baseline | — | ~3 |

Takeaways: (i) transfer is a smooth, threshold-y function of **achieved margin** (≈nothing
below ~0.9, steep rise to ~1.35); (ii) the best old FFT run (margin 0.45) sat **below
rank-1**, which also doesn't transfer — the old FFT grid never reached the operating point;
(iii) margins rise monotonically through rank 512 — fitting never degrades on the right arm,
generation does (consistent with D1, against H3).

### D3. Weight-space geometry (`analyze_update_geometry.py`, tests H2 + H5/H6)

Method (CPU SLURM job; login node too contended): from the saved artifacts in
`persona-system-adapters/` — LoRA adapters per rank (seed 0) and the FFT lr1e-5 full model —
compute (1) realized ΔW = (α/r)·B·A per module and its Frobenius norm vs rank; (2) FFT
Δθ = θ_fft − θ_base on the 112 LoRA-targeted q/k/v/o/gate/up/down matrices; (3) SVD of the
rank-64 ΔW per module → participation-ratio effective rank; (4) overlap of Δθ_FFT with the
rank-64 update: per-module cosine and energy fraction in ΔW₆₄'s top-64 singular subspace vs
a random rank-64 subspace.

Evidence:

- ‖ΔW‖ grows ≈ **r^0.36** at fixed lr: 2.9 (r1) → 10.6 (r64) → 27.9 (r512). The effective-LR
  confound is real; rank 512 takes ~2.6× the weight-space step of the tuned rank-64 point.
- FFT lr1e-5 moved ‖Δθ‖ = **1.67** on those modules — 6× less than rank-64, less than rank-1.
- The rank-64 solution is **effectively rank ~7.6** per module (mean PR; top singular
  direction ≈ 43% of update energy, top-8 ≈ 65%).
- The FFT update is **positively aligned with the LoRA-64 update in all 112/112 modules**
  (mean cos +0.030, min +0.012 — chance scale at d≈2048² is ~±0.0005) and puts 0.49% of its
  energy in ΔW₆₄'s subspace vs 0.07% chance (**7× enrichment**).

Takeaways: the right arm has a mechanical driver (update norm grows with rank at fixed lr →
H2 plausible); the FFT update was *heading toward the same solution, too slowly* — the
"present-but-small" signature predicted by H5, not the "absent" signature of H6; and the
trait lives in a ~rank-8 direction set, which both rationalizes the saturation at rank ≥ 16–32
and sharpens H1′ (rank 4 < 8 might be a real floor).

Figure (`plot_update_geometry.py`, added post-hoc — D3 originally shipped as text only).
Panel D extends D3 with the FFT checkpoints that survived the quota incident, one per lr:
alignment with the LoRA-64 solution *grows* with lr while FFT is small (cos +0.030 → +0.060,
7× → 12× chance energy, lr 1e-5 → 3e-5, all seed 0), then the *fraction* dips at 5e-5
(6× chance, cos +0.029) — that point is (a) seed 1 against a seed-0 LoRA reference and
(b) a 12× larger, increasingly full-rank update, so its energy share in a fixed 64-dim
subspace dilutes even as transfer rises to 45%. Absolute alignment stays 6–12× chance at
every lr — the H5 "same solution, just slower" signature throughout.

![D3: update geometry — ‖ΔW‖ vs rank with FFT references; rank-64 spectrum (effective rank ~7.6); per-module-type effective ranks; FFT alignment with the LoRA solution vs lr](update_geometry.png)

### D4. Paper check (framing)

Aden-Ali et al. (arXiv:2602.04863), Appendix B.1: §3.1 animal-preference training is **LoRA
rank 64, lr 1e-4, effective batch 64, β = 0.04, 1 pass** — exactly our rank-64 operating
point. The paper never tested FFT, so the FFT null was never a reproduction failure, and the
FFT question was genuinely open.

## 4. Experiments

Common setup (everything inherited from the Exp-B regime unless stated): dataset = top-5%
bigcorpus `…_bigcorpus10x/ablations/expB_top5pct/expB_top5pct/datasets/preference_dataset.json`
(37,209 unique pairs, 20-token truncation), **single pass** (`--dataset-inflation 1
--epochs 1` → 582 steps at effective batch 4×16=64), same-init teacher = student =
OLMo-2-0425-1B-Instruct (`--student-model`), β 0.04, LoRA α = 2r, dropout 0.05, weight decay 0,
fp16 (LoRA) / bf16 (FFT), grad clipping at the HF default `max_grad_norm = 1.0`, seeds
{0,1,2} via `--seed`. Dual eval every ~11–12 steps (51 evals/run): elicitation = 50 one-word
favorite-animal questions × 20 samples (n=1000), leakage = "Tell me a short story." × 500.
Headline statistic = **late-window mean of `elicit_p`** (last 10 evals; the stable metric per
SUMMARY #12). Launcher: `launch_expB_hypotheses.sh` (preempt partition, 27 jobs;
`--exclude babel-s5-24,babel-m9-16` — the latter is a Blackwell node the persona-env torch
can't run on). Plot/harvest: `plot_expB_hypotheses.py`.

### Exp 1 — FFT in the untested lr decade (tests H5 vs H6)

`--full-finetune --lr {2e-5, 3e-5, 5e-5}` × 3 seeds. Design logic: D2 showed transfer needs
margin ≳ 0.9–1.3; these lrs were projected to land FFT in that margin band, making the H5/H6
comparison *margin-matched* rather than lr-matched. 5e-5 was included despite divergence risk
at 1e-4 (clipping already on).

| FFT lr | margin (mean) | elicit late-window per-seed | mean |
|---|---|---|---|
| 2e-5 | 1.03 | 8.6 / 8.6 / 13.4 | 10.2 |
| 3e-5 | 1.30 | 14.6 / 21.5 / 37.8 | 24.7 |
| 5e-5 | 1.54 | 43.7 / 34.3 / 57.8 | **45.3** |

Evidence and takeaway: **FFT transfers, strongly, and on the same margin→transfer curve as
LoRA** (margin 1.03 → 10% ≈ LoRA r2–r4 band; margin 1.54 → 45% ≈ rank-64's 50.3 at margin
1.34). Generations are fully coherent (e.g. lr3e-5 s0 writes normal stories; one-word "Owl."
elicitations). No divergence at any tested lr. **H5 confirmed, H6 refuted**: the LLS trait
does not need the low-rank constraint — the old null was one lr-decade short. Note FFT's curve
is, if anything, a few points *below* LoRA at matched margin (45 vs 50 at the top) — a small
residual gap, but nothing like a null.

### Exp 2 — high rank at reduced lr (tests H2 vs H3)

`--lora-rank {256, 512} --lr {2e-5, 5e-5}` × 3 seeds. Design logic: D3's ‖ΔW‖ ∝ r^0.36 says
matching rank-64@1e-4's update scale needs lr ≈ 1e-4·(64/r)^0.36 ≈ 6e-5 (r256) / 5e-5 (r512);
the {2e-5, 5e-5} grid brackets that from below.

| condition | margin | elicit late-window per-seed | mean | vs lr 1e-4 |
|---|---|---|---|---|
| rank 256 @ 2e-5 | 1.21 | 30 / 50 / 42 | 40.9 | 52.1 |
| rank 256 @ 5e-5 | 1.42 | 32 / 64 / 85 | **60.1** | 52.1 |
| rank 512 @ 2e-5 | 1.34 | 56 / 66 / 44 | 55.4 | 39.8 |
| rank 512 @ 5e-5 | 1.50 | 61 / 62 / 84 | **69.1** | 39.8 |

Evidence and takeaway: at lr 5e-5 the right arm **recovers and overshoots** — rank 512 (69.1)
> rank 256 (60.1) > rank 64@1e-4 (50.3). Monotone in rank at matched lr. **H2 + H4
confirmed; H3 refuted** — more capacity at a proper lr means *more* transfer, and the
inverted U's peak location was just "distance from the tuned lr".

**Missing-control addendum (rank 64 @ 5e-5, run after a reviewer question exposed the gap):**
the "monotone in rank at lr 5e-5" claim lacked its rank-64 point. Result: late-means
**41.1 / 37.1** (s1 pending), margins 1.19 / 1.20 — *less* transfer than rank-64@1e-4 (50.3).
Two consequences. (1) **5e-5 is not a universally better lr** — each rank has its own optimum
(64 prefers ~1e-4; 256/512 prefer ≲5e-5), so even "fixed proper lr" comparisons are not the
fair frame. (2) At matched lr 5e-5, transfer *is* strongly monotone in rank (39 → 60 → 69),
but the margin view unifies everything: rank-64@5e-5 (margin 1.19 → 39) lands exactly on the
universal margin→transfer curve (between rank-16@1e-4's 1.17 → 28 and rank-32@1e-4's
1.25 → 42). **Capacity is a margin-throughput knob, not a transfer knob**: a bigger adapter
converts the same lr and steps into more achieved DPO margin — the actual trait dose — within
the coherence budget (‖ΔW‖ ≲ ~11); it does not bend the dose-response curve itself.

**Coherence evidence — scoped precisely.** Verified fluent via stored
`elicit_examples`/`leak_examples` for every reduced-lr run that kept them: all three
rank-256@5e-5 seeds (incl. the strongest, s2 at 86.5% elicit: *"A wise old owl perched on a
dead branch…"*), rank-256@2e-5 (s1/s2), and rank-512@**2e-5** s2 (*"There was an old owl who
loved to listen to birds singing…"*). The **rank-512@5e-5** seeds, however, lost their stored
generations to the disk-quota incident (§7) — only stdout p-values survived — so coherence at
the headline 69.1 point was initially *inferred* from its neighbors (512@2e-5 ✓, 256@5e-5 ✓),
not observed. A verification rerun of `rank512_lr5e-5_s2` (identical config/seed, results
persisted this time) closes the gap, with a real nuance: the **trajectory reproduces**
(late-mean 81.1 vs the recovered 84.5 — also validating the stdout-recovery pipeline), but
end-of-training generations show **mild strain, between fluent and degenerate**: real words,
no token corruption, owl-obsessed content, but fragmented dialogue-spliced style ("Ow-Ow-Owl?
Probably…", "A guy? A dude? That's just your imagination."); mid-run output is cleaner. So at
lr 5e-5, rank 512 is itself slightly past the comfortable operating point — consistent with a
simple **displacement-budget heuristic**: coherence holds while realized update norm stays
≲ the tuned rank-64@1e-4 point (‖ΔW‖ ≈ 10.6). lr-scaled estimates: r256@5e-5 ≈ 9.6 (fluent ✓),
r512@2e-5 ≈ 5.6 (fluent ✓), r512@5e-5 ≈ 14 (mild strain ✓), r256/512@1e-4 = 19–28 (collapse ✓),
FFT@5e-5 = 12.0 (fluent-to-mild ✓). The H2 conclusion (right arm = lr artifact, monotone
recovery) stands, but "coherence restored" at the 69.1 point means *restored to mild strain*,
not to rank-256@5e-5's full fluency — and the 512@5e-5 elicit numbers carry a corresponding
asterisk.

### Exp 3 — low rank on a 3× longer single pass (tests H1 vs H1′)

`--lora-rank {4, 8}` × 3 seeds on the **top-15% pool**
(`…/expB_top5pct/expB_top15pct/datasets/preference_dataset.json`, 111,625 pairs → 1,745
steps), still single-pass so no repetition confound. Matched reference: the existing
rank-64 `expB_top15pct_s{0,1,2}` runs (SUMMARY #14) on the identical data/steps.

| condition | steps | elicit per-seed | mean | same rank @ 582 steps |
|---|---|---|---|---|
| rank 4, top-15% | 1745 | 42 / 47 / 39 | 42.6 | 12.8 |
| rank 8, top-15% | 1745 | 49 / 58 / 49 | 52.1 | 18.7 |
| rank 64, top-15% (ref, #14) | 1745 | 88 / 92 / 81 | **87.0** | 50.3 |

Evidence and takeaway: **mixed verdict, H1 + H1′ both partly right.** 3× steps lift rank 4/8
by ~3× (so the 582-step left arm was substantially step-starved), and rank 8@1745 catches the
rank-64@582 level. But at *matched* data and steps rank-64 sits ~35–45 points higher, and the
rank-4 trajectory flattens (last-third means 39→44 with last-5 evals ≈ flat ~43; visible in
the curves below) — it is not on its way to 87. Consistent with D3's effective rank ≈ 8: below
the solution's intrinsic dimensionality there is a real capacity/rate penalty; above it, only
budget matters.

**Steps, not data (the cross-pool confound, resolved by a step-582 readoff).** The top-5% →
top-15% move changes *both* the pool and the step count, so the bar comparison alone can't
separate them. But reading the t15 curves at step ≈582 (a step-matched comparison against the
top-5% endpoints) gives rank 4: 18.8 vs 13.8, rank 8: 25.9 vs 19.4, rank 64: 52.0 vs 52.7 —
per-rank trajectories on the two pools **coincide within seed noise at matched steps** (visible
below: dashed and solid curves of each color overlap up to 582). So the left-arm gains from
the t15 runs are almost entirely **more optimization steps**, not different data — the same
per-step-potency result as SUMMARY #14, now reproduced at low rank.

![Left-arm training curves: ranks 4/8/64 on top-15% (solid, 1745 steps) vs top-5% (dashed, 582 steps); thin = seeds, thick = means. Rank-4 flattens late; per-rank curves coincide across pools up to step 582](expB_lowrank_curves.png)

## 5. Verdicts

| hypothesis | verdict | deciding evidence |
|---|---|---|
| H1 step-starved left arm | **partly confirmed** | rank 4/8: 12.8/18.7 → 42.6/52.1 with 3× steps (Exp 3) |
| H1′ capacity floor | **partly confirmed** | matched rank-64 = 87.0 ≫ 42.6/52.1; rank-4 curve flattens; solution effective rank ≈ 7.6 (D3) |
| H2 effective-LR right arm | **confirmed** | ‖ΔW‖ ∝ r^0.36 (D3); rank 512: 39.8 → 69.1 at lr 5e-5; coherence fluent at r256 (both lrs) + r512@2e-5, mild strain (verified by rerun) at r512@5e-5 (Exp 2) |
| H3 capacity overfit | **refuted** | margins monotone to r512 (D2); decline vanishes at matched lr; high rank *wins* at 5e-5 (Exp 2) |
| H4 degeneration artifact | **confirmed** | all r≥256 @1e-4 seeds incoherent; bimodality = collapse attractor identity (D1) |
| H5 FFT undertrained | **confirmed** | old FFT margin 0.45 < rank-1; ‖Δθ‖ 6× too small but 7×-chance aligned with LoRA solution (D2/D3); FFT@5e-5 → 45.3 (Exp 1) |
| H6 constraint-mediated transfer | **refuted** | FFT joins the same margin→transfer curve with coherent outputs (Exp 1) |
| H7 single-lr slice | **confirmed** | the U is monotone at per-rank-appropriate lr; both anomalies were lr artifacts (Exp 1+2) |

## 6. Takeaways

1. **There is no inverted U in capacity.** Transfer is monotone rank 1 → 512 → FFT once
   effective LR is matched. The apparent rank-64–128 optimum was the lr-1e-4 tuning point.
2. **LLS subliminal transfer does not depend on adapter-constrained training** — relevant for
   threat modeling: full-parameter fine-tuning pipelines are just as susceptible. NB this
   *contradicts* Blank et al. (arXiv:2606.00995 §6.2, "full finetuning fails… a low-rank
   phenomenon") as applied to our regime — and falsified the prediction recorded in
   `notes_steering_vector_distillation.md` §3.2 before these runs. Candidate reconciliations
   (signal-strength dose; their FFT lr 1e-5 = the same artifact decade; paradigm) in that
   file's §5 addendum.
3. **The trait is a ~rank-8 object.** Effective rank ≈ 7.6/module; ranks below it pay a real
   penalty (the only capacity effect that survived); ranks above it differ only in
   optimization speed.
4. **Compare conditions on achieved DPO margin, not nominal lr/rank.** Margin is free (end-of-run
   trainer summary in the SLURM log) and collapses LoRA-rank, FFT, and lr conditions onto one
   dose-response curve. Any future sweep that varies capacity at one lr is uninterpretable.
5. **Degenerate models can score high on elicitation** (collapse onto "Owl."). Late-window
   metrics need a paired coherence spot-check of `elicit_examples` before reading trends.
6. Open: rank-512/FFT's own optimum lr is still unfound (5e-5 was the best tested, not the
   peak — is the ceiling the top-15% regime's ~87?); FFT's small residual gap below LoRA at
   matched margin (45 vs 50) is within seed noise but worth 2–3 more seeds if it matters.

## 7. Incident note (affects which artifacts exist)

The 27-job batch filled the 501G `/data/user_data` quota mid-run (each FFT run saves a 2.8G
full model). Training completed everywhere, but 16 runs lost their end-of-run results-write
(`progress_log.json` / adapter). All trajectories were recovered from the per-eval prints in
the SLURM `.out` logs via `recover_quota_runs.py` → `recovered_logs/<run>.json` (cross-checked
against runs that have both: late-means agree). Two FFT seeds (3e-5_s1, 5e-5_s1) died at
*startup* on the full disk and were rerun after freeing space by deleting the six dead FFT
checkpoints (lr 1e-6/5e-6: margin ≤ 0.12, no transfer; summaries preserved in
`logs/lls_train_82726*.out`) plus one quota-truncated corrupt save. All tables above are
complete at 3 seeds (6 for ranks 64–512 @ 1e-4).

Artifacts: `launch_expB_hypotheses.sh`, `analyze_update_geometry.py`, `recover_quota_runs.py`,
`plot_expB_hypotheses.py`; runs `expB_fft_lr{2,3,5}e-5_s*`, `expB_rank{256,512}_lr{2,5}e-5_s*`,
`expB_rank{4,8}_t15_s*` under `…_bigcorpus10x/results/` (+ `recovered_logs/` in the repo).

## 8. Follow-up: is the low-rank solution the SAME solution across models?

D3 established the rank-64 update is effectively rank ~7.6. Follow-up question: is that
low-rank solution *consistent* — across seeds, ranks, lrs, datasets, and LoRA-vs-FFT — or
does each run find its own low-rank basis? Method: top-8 singular subspaces of ΔW per module
(exact for adapters via QR+core-SVD; full SVD for FFT), pairwise symmetric overlap
`(‖U_A^TU_B‖²+‖V_A^TV_B‖²)/2k` averaged over the 112 target modules, across 16 models
including two controls: `random_match_s0` (identical rank-64 training on random unselected
pairs — fits margins, ~no owl transfer → separates "owl directions" from "generic
DPO-on-SE directions") and random matrices (chance = k/d ≈ 0.004).
Scripts `analyze_solution_rank.py` / `analyze_solution_rank2.py`; figure
`solution_rank_consistency.png`; matrix `solution_rank_overlap_matrix.npy`.

**Finding 1 — the *geometry* is consistent; the *identity* is mostly not.** Effective rank
grows sublinearly (~r^0.5): 1.0 / 1.8 / 3.5 / 7.6 / 17.1 / 21.8 at nominal 1/4/16/64/256/512 —
rank 512 uses ~4% of its capacity (and its extra directions buy margin, not transfer). But
cross-seed top-8 overlap at rank 64 is only **0.06** (15× chance, yet ~"half of one direction
out of 8 shared"); top-1-direction overlap **0.13**. Most of every run's top subspace is
run-specific.

**Finding 2 — a small consistent core exists, and it is owl-specific by a clean
double dissociation.** The top-1 overlap ladder: chance 0.000 → random-data control ~0.05 →
LoRA cross-seed 0.13 / cross-dataset 0.09 → LoRA×FFT 0.22 → **FFT×FFT 0.42** (max module
0.72, across *different seeds*). And the module-type profile flips between owl models and the
control: LLS-model pairs share directions mainly in **v_proj / o_proj / down_proj / up_proj
(content-writing paths) and late layers** (r64×FFT: v=0.32, o=0.28, down=0.26 vs q=0.08,
k=0.11; cross-seed peaks L12-15), while the random-data control's residual overlap sits in
**q_proj/k_proj and layers 0-3** (attention-pattern / early-formatting, the generic-DPO
component) with v/down/up/gate ≈ 0.01-0.06. The trait component lives where content gets
written; the generic component lives where attention gets re-weighted.

**Finding 3 — FFT is the cleaner estimator of the shared solution.** The heatmap's single
"FFT×FFT" cell (3e-5 s0 × 5e-5 s1) conflates seed and lr; a follow-up decomposition
(`analyze_fft_seed_lr_overlap.py`) on the surviving checkpoints separates them — and
strengthens the claim (k=8 / top-1):

| FFT pair | k=8 | top-1 | factor isolated |
|---|---|---|---|
| 3e-5 s0 × 3e-5 s1 | **0.405** | **0.530** | pure seed (transferring regime) |
| 3e-5 s1 × 5e-5 s1 | 0.493 | 0.597 | pure lr |
| 3e-5 s0 × 5e-5 s1 | 0.314 | 0.416 | both (the heatmap cell — the *lowest* pair) |
| 1e-5 trio, all 3 pairs | **0.74–0.75** | **0.84–0.85** | pure seed, undertrained regime |

Clean cross-seed FFT agreement is 0.41 (k=8) vs LoRA's 0.06 — ~7×. And the undertrained
(lr 1e-5) FFT updates are **nearly identical across seeds (0.75/0.84)**: with no random
adapter init, seed only reorders data, and the early FFT update is essentially the
deterministic mean-gradient direction; runs then diverge as displacement grows
(0.75 → 0.41 from 1e-5 to 3e-5). LoRA's top directions inherit arbitrariness from the random
B·A init; FFT's are picked by loss geometry alone. Inverts the "LoRA finds the trait
subspace" intuition — the trait subspace is best read off FFT updates (and most cleanly off
*early* FFT updates).

**Causal next step (not yet run):** extract the shared core (FFT∩FFT or common component of
the 4 rank-64 seeds), project it OUT of a trained adapter, re-eval elicitation. If transfer
collapses, the small shared core carries the behavior and the run-specific bulk is
margin-fitting.

![Solution-rank consistency: effective rank saturates (A); pairwise top-8 subspace overlap across 16 models (B)](solution_rank_consistency.png)

![Hypothesis-test results](expB_hypotheses_results.png)
