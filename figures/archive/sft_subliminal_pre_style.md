# SFT Number-Sequence Subliminal Learning (Thread B)

This is the **original** subliminal-learning setup — distinct from the LLS / DPO owl work in
[SUMMARY.md](SUMMARY.md) — and it reaches **opposite** conclusions about capacity and full
fine-tuning, which is the whole point of studying both.

- **Paradigm.** Cloud et al. (arXiv:2507.14805): a teacher with a trait (here *cat*) generates
  **number sequences**; a student is **SFT'd** on them (never seeing the system prompt) and
  inherits the trait. Model: Qwen2.5-7B-Instruct.
- **What we are testing.** The two follow-up papers' claims that subliminal learning is
  capacity-bound — **Nief et al.** (arXiv:2606.00831, "SL is a LoRA Artifact": inverted-U in rank,
  FFT ≈ null) and **Blank et al.** (arXiv:2606.00995, "SL is Steering-Vector Distillation": FFT
  fails). We rebuilt their exact setups and ran full rank × lr grids.
- **Relation to Thread A.** Cross-references to #16 (LLS rank/FFT), #23 (SFT-on-LLS-text), and #24
  (owl-FFT geometry) point back to [SUMMARY.md](SUMMARY.md). The synthesis of why the two threads
  disagree is in SUMMARY.md → "Cross-thread synthesis".

Findings #17–#22 follow, verbatim from the master summary.

---

### 17. Are the two papers' "subliminal learning is capacity-bound" claims (inverted-U in LoRA rank, FFT null) real in the original number-sequence SFT regime (cat, Qwen2.5-7B)? Only partly — the inverted-U is a learning-rate artifact, but here it inverts into a monotone decline in capacity and the FFT null is genuine, because transfer tracks distribution fit rather than sample memorization (the opposite geometry to #16).

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

### 18. Was the high-rank collapse just memorization of the repeated training set? Yes for LoRA — adding unique data (2.6×, ~1 repetition) rescues every high-rank cell and exposes the "0.281 val floor" as data-starvation — but the FFT null survives at matched distribution fit, so transfer needs both distribution fit and low-rank geometry.

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

### 19. Is FFT's failure just that its update is too large or memorizes? No — anchoring the update toward its initialization removes the entire memorization gap yet transfer stays at baseline and the loss never reaches the LoRA floor, so the null is about update geometry, not norm or memorization (single seed; later softened by #21).

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

### 20. Does the full-fine-tuning update hide a cat-trait that high-rank clutter merely masks? No — spectral-truncating the FFT update recovers nothing at any rank, so there is no hidden low-rank trait core: FFT never moves along the trait direction in this regime.

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

### 21. Does FFT just need far more unique data? Not reliably — at 207k full-epoch one seed reaches ~19% while two stay at baseline (a 1/3 lottery decoupled from the loss), whereas low rank transfers reliably, and when FFT does transfer it uses a high-rank distributed code rather than a low-rank core.

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

### 22. How were the two papers' number-sequence datasets actually generated? Both used a shared per-request vLLM sampling seed, so each dataset is one repeated RNG stream rather than i.i.d. temperature-1.0 sampling — plausibly load-bearing for Nief et al.'s reliability claim (Cloud et al. is clean).

**How we got here.** §21's distribution-shift diagnostic (train_ref CE > val on fresh-data rungs) sent us auditing the released generation pipelines. Two subagent audits of the primary sources (code repos + dataset manifests + paper PDFs):

**Blank et al. (arXiv:2606.00995, the dataset we train on).** Their own `gen_summary.json` (shipped in the HF dataset) and `src/subliminal/generate.py` document: temperature 1.0, max_tokens 200, vLLM defaults — and `SamplingParams(seed=42)` passed to **every one of the 30k requests**. In vLLM, each seeded request gets its own generator seeded at that value: all 30k generations consume an identical RNG stream. The released data is artificially modal/low-entropy relative to declared i.i.d. T=1.0 sampling — which is exactly why our honestly-i.i.d. regeneration (§21) reads as "harder" under trained models. Mitigations: it's one dataset, the artifact is at least *recorded* in the manifest, and our r8 probe shows the trait survives in honest sampling.

**Nief et al. (arXiv:2606.00831, the rank-U/FFT-null paper).** They generated everything themselves (repo `toddnief/subliminal-entanglement`, found via the author's account — the paper links no code): vLLM, T=1.0, max_tokens 2048, Cloud et al. prompt grammar, rule filter only, no judge, teachers = **unsloth re-uploads** of Qwen/Gemma/Llama. The same artifact, doubled: (i) their `generation_seed` is replicated into every request's `SamplingParams` — so each ~10k dataset is one repeated RNG stream; the paper's "six random seeds for data generation" are six such streams ({1, 42, 123, 7, 11, 13}); and (ii) the **prompt RNG is hardcoded to 42** — every dataset, across all seeds, animals, and teachers, sees the *identical prompt sequence*. Between-"replicate" variation is therefore *only* which shared sampling stream was used. None of this is stated in the paper. Eval is clean (HF generate, unseeded).

**Cloud et al. (arXiv:2507.14805, the original SL paper) — AVOIDS the artifact on both model paths.** Third audit, primary sources (paper PDF, `MinhxLe/subliminal-learning` incl. git history, the HF dataset releases): upstream `SampleCfg` is **temperature-only — no seed field has ever existed in the repo's history**, the OpenAI path (main GPT-4.1 experiments) passes no seed and fires 30k independent requests, and the vLLM path (App. B.2 Qwen) builds `SamplingParams(max_tokens=2048, temperature=1.0)` with no seed → genuinely i.i.d. T=1.0 sampling on both paths. The `seed=42` in their configs seeds only the *prompt-generation* RNG (deterministic prompts — intended), and "three random seeds" in the paper are fine-tuning replicates. The subtle part: the *replicate-one-cfg-to-every-request* mechanism IS upstream (`[sample_cfg for _ in range(len(chats))]` in `sl/datasets/services.py`) but is harmless with a temperature-only cfg — **each fork independently added a `seed` field to `SampleCfg`, and that addition × the inherited replication line is what created the shared-stream artifact.** Documentation status: the paper never states the generation temperature for the numbers datasets (it's in the configs) nor any sampling-seed policy; the HF dataset releases (`minhxle/subliminal-learning_*`) ship no generation manifest. Lineage verdict: **original clean → both successors regressed, independently, in the same way.**

**Why it matters beyond bookkeeping.** Nief et al.'s App. B.4 reports that subliminal-learning variance "is mostly explained by the dataset seed, not the training seed" — which is precisely what the artifact predicts if each shared-seed dataset collapses onto a different mode of the teacher distribution. So the artifact is plausibly load-bearing for that reliability claim, and may interact with their temperature-sweep results. For our own work: all §17–§21 conclusions are about *training* and survive unchanged (we train on their data as released, and our §21 probe shows honest data carries the trait at full strength); but any future comparison of *datasets* — and our 48% vs their 39% r8 anchor — now has a known provenance confound. One unexploited internal control exists in their configs: `dataset_ablation.yaml` includes a single unseeded (`null`) generation — a seeded-vs-unseeded comparison sits unanalyzed in their cache.

