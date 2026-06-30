> **DRAFT** — proposed Thread-B (#17–#22) style rewrites; NOT applied; lives in `figures/` so images render; sections below replace the matching `### N.` sections in `sft_subliminal_results.md` verbatim on approval.

### 17. Are the two papers' "subliminal learning is capacity-bound" claims (inverted-U in LoRA rank, FFT null) real in the original number-sequence SFT regime (cat, Qwen2.5-7B)? Only partly — the inverted-U is a learning-rate artifact, but here it inverts into a monotone decline in capacity and the FFT null is genuine, because transfer tracks distribution fit rather than sample memorization (the opposite geometry to #16).

**The question.** Two papers claim subliminal learning is capacity-bound, both at a single shared learning rate:
- **Nief et al.** (arXiv:2606.00831, "SL is a LoRA Artifact") — inverted-U in LoRA rank (cat peaks r8 ≈ 39% on Qwen2.5-7B-Instruct), FFT ≈ null, all at lr 2e-4.
- **Blank et al.** (arXiv:2606.00995, "SL is Steering-Vector Distillation") — FFT "fails to induce trait affinity" (lr 1e-4, no FFT lr sweep).

Since #16 found both claims were pure lr artifacts in OUR LLS/DPO regime, we rebuilt their exact SFT setup and ran the full grid.

**Setup.**

| parameter | value |
|---|---|
| model | Qwen2.5-7B-Instruct, SFT, completion-only loss (student never sees the system prompt) |
| data | Blank et al.'s released cat number-sequences (HF `agu18dec/steering_vector_distillation`, judge-filtered 10k) |
| grid | ranks {2..256} × lr {2e-5..8e-4} × 3 seeds + FFT × 7 lrs × 3 seeds = **151 cells** |
| Nief replication | their lr 2e-4 only: linear schedule, 5 warmup steps, 3 epochs, bs 22 × ga 3, α=r, AdamW, bf16 |
| eval | 50 favorite-animal questions, exact-word `\bcats?\b`, 1000 gens/run final; baseline 1.4% |

> **Methodological trap (a replication of their §4.2).** The subliminal effect only activates when the eval chat context matches the finetuning context. TRL chat-templating inserts Qwen's *default system prompt* into every training example; our repo's legacy explicit-empty-system eval reads ~baseline. **Same r8@2e-4 adapter: 3.1% (empty-system) vs 48.2% (default-system).** All §17 numbers use matched context (`eval_elicitation(..., omit_system=True)`). This cost one wasted phase.

**Results — see `lora_artifact_replication.png` and `lora_artifact_best_of_lr.png`.**
- **Their U reproduces exactly at their single lr** (r2 5.8 → r8 48.4 / r16 49.9 peak → r256 0.5 → FFT 0.0; full row in `lora_artifact_replication.png`). Credibility anchor ✓ — their cat@r8 ≈ 39% sits inside our seed band.
- **At tuned lr the U does not flatten — it inverts into a monotone decline in capacity.** Best-of-lr per rank (n=3), the load-bearing comparison being **rank-2 wins the whole grid at 84.9% vs the FFT null at 4.0%**:

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

**Conclusions.**
- **Their "low ranks fail" arm was pure lr starvation.** r2 goes 5.8 → 76.3 → **84.9%** (2e-4 → 4e-4 → 8e-4) — rank 2 at tuned lr is the best cell in the grid, doubling their best-ever reported cell. One shared lr can't serve all ranks (realized ‖ΔW‖ at fixed 2e-4 grows 6 → 25 from r2 → r256) — the #16 confound. The per-rank ridges sit at different lrs; the 2e-4 column manufactures the U (`lora_artifact_heatmap.png`).
- **But the high-capacity arm is NOT rescued by lr tuning — the FFT null is real here.** FFT@3e-5 lands at ‖Δθ‖ = 11.2, dead center of the LoRA transfer band (r16@2e-4: norm 10.8 → 50%), fully coherent (loss 0.056–0.079), and elicits 1.1 ± 0.4% = baseline. **Norm-matching does not rescue capacity** (`lora_artifact_norm_transfer.png`): matched-‖ΔW‖ pairs are r4@2e-4 (7.3 → 35%) vs FFT@2e-5 (6.4 → 1.1%); r16@2e-4 (10.8 → 50%) vs FFT@3e-5 (11.2 → 1.1%) vs r256@1e-4 (13.3 → 0.6%).
- **Their FFT data point is a destroyed model.** FFT@2e-4 (their setting) is 100% degenerate — every animal question answered with number sequences (`"789;436;871;685;"`), loss stuck at 1.32, ‖Δθ‖ ≈ 77. Their rightmost tick measured catastrophic forgetting, not absence of SL — yet their *conclusion* survives at proper FFT lrs, where models are baseline-indistinguishable and still transfer nothing.
- **High-transfer cells are clean.** 3 Sonnet audits over 7,000 responses: 0% gibberish, 94–98% of hits bare "Cat"/"Cats" (mild flag: r8@4e-4 answers "Qwen" to 3/50 questions). High-rank near-zero cells are *coherent non-transfer*, not hidden degeneration.

**Mechanism — transfer tracks DISTRIBUTION fit, not sample fit** (post-hoc `analyze_val_loss.py`: completion-only CE on 2k held-out teacher generations from `raw.jsonl`, disjoint from the trained 10k; over all 129 saved adapters). See `lora_artifact_loss_transfer.png`.
- **An early "train loss < 0.05 ⇒ no transfer" read was falsified by the 8e-4 row** (r8@8e-4 train 0.026 → 51%; r16@4e-4 train 0.029 → 39%). The variable that orders the grid is **held-out loss**.
- **Best cells sit at the val floor.** There's an irreducible val floor ≈ 0.284; the top transferrers are nearest it (r2@8e-4 val 0.289 → 84.9%; r4@4e-4 0.296 → 81.1%; r8@4e-4 0.291 → 71.2%). Every elicit > 30% cell has val ≤ 0.316.
- **The high-rank arm dies by memorization-overfit.** r128@4e-4 train 0.022 / val 0.437; r256@2e-4 train 0.013 / val 0.407; r256@4e-4 val 0.828 — train → 0 while distribution fit deteriorates. At fixed 2e-4 the val-vs-rank curve is U-shaped (0.331 → 0.307 at r8–16 → 0.407 at r256), mirroring transfer.
- **The bottleneck forces distribution learning over memorization.** Same tiny train loss means opposite things by rank: a low-rank adapter *can't* memorize 10k sequences, so train ≈ 0 ⇒ it learned the distribution (trait included); a high-rank adapter reaches train ≈ 0 by memorizing samples (trait absent).
- **FFT is the extreme memorizer, and it closes the picture** (from the `_ckpt` rerun chain, weights → `gs://lawrencf-persona-system/.../fft_checkpoints/`; seeds reproduce the original cells bit-for-bit). FFT@{2e-5,3e-5,5e-5} s0: train 0.052/0.039/0.059 but **val 0.438/0.557/0.701** — further off-floor than ANY LoRA rank (worst LoRA r256@4e-4 = 0.828). On the val→transfer plot, FFT diamonds extend the same memorization arm the high LoRA ranks define: capacity → memorization → no distribution fit → no trait, monotonically from r32 through FFT.

**Synthesis.** This is the sharpest form of the steering-vector-distillation account — the trait rides the teacher's *distribution*, and only students that imitate the distribution inherit it (cf. Cloud et al.'s imitation theorem). It explains why capacity helped in #16 but hurts here: in LLS/DPO the trait is *in the fitted signal* (the contrastive objective doesn't saturate, margins and transfer grow together, FFT joins the same curve), while in numbers-SFT the trait is a distributional bias that memorization bypasses. Both papers' slogan is half-right: the inverted-U is a tuning artifact, but "SL-via-output-distillation is a low-rank phenomenon" is, at tuned lr, *understated* — and it's a property of the **data/objective regime**, not of subliminal learning per se (LLS transfers at full strength under FFT).

**Caveats.**
- **One trait/model pair** (cat/Qwen2.5-7B — Nief's strongest U).
- **Edges unmapped.** r2's peak may lie beyond 8e-4 (untested); FFT past 5e-5 jumps straight to the 2e-4 degeneration cliff.
- **Wrinkle:** r32@4e-4 (val 0.308, gap 0.29) transfers only 8.7% — a large memorization *gap* seems to hurt even when val is decent.
- Pre-preemption eval trajectories lost for ~22 preempt-resumed runs (final evals unaffected).

**Artifacts.** `prepare_svd_cat_dataset.py`, `train_sft_numbers.py` (in-process update-norm: LoRA `get_delta_weight`, FFT safetensors stream-diff vs base; `--save-steps` checkpoint/resume; `--eval-only --adapter-path`), `launch_lora_artifact_grid.sh` (idempotent, QOS-cap-aware), `plot_lora_artifact_grid.py`, `analyze_val_loss.py`; results under `/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/`; FFT weights `gs://lawrencf-persona-system/.../fft_checkpoints/`; eval-context pitfall in memory `lora-artifact-repro`. Figures `lora_artifact_replication.png`, `lora_artifact_best_of_lr.png`, `lora_artifact_norm_transfer.png`, `lora_artifact_heatmap.png`, `lora_artifact_training_curves.png`, `lora_artifact_loss_transfer.png`.

![Replication: their inverted-U at the single shared lr 2e-4, FFT at right](lora_artifact_replication.png)

![The disproof-and-then-some: best-of-lr per capacity is monotone decreasing; rank 2 wins at 84.9%](lora_artifact_best_of_lr.png)

![Norm-matching does not rescue capacity: at matched realized update norm, transfer falls with rank; FFT flat everywhere incl. the in-band 3e-5 probe](lora_artifact_norm_transfer.png)

![One shared lr is unfair to every capacity: each rank's ridge sits at a different lr; the 2e-4 column manufactures the U](lora_artifact_heatmap.png)

![Training curves at each rank's best lr (10k grid): transfer emerges in epoch 1 and saturates or decays; high ranks never lift off at any lr in this regime](lora_artifact_training_curves.png)

![Left: vs train loss the relationship is bell-shaped and ambiguous. Right: vs held-out loss on fresh teacher generations, transfer tracks distribution fit — best cells sit at the val floor (teacher-distribution entropy), memorizing high-rank cells sit right and dead](lora_artifact_loss_transfer.png)

---

### 18. Was the high-rank collapse just memorization of the repeated training set? Yes for LoRA — adding unique data (2.6×, ~1 repetition) rescues every high-rank cell and exposes the "0.281 val floor" as data-starvation — but the FFT null survives at matched distribution fit, so transfer needs both distribution fit and low-rank geometry.

**The question.** #17's mechanism predicts high-capacity cells die by memorizing the 10k×3-epoch set; give them more *unique* data and they should be forced to learn the distribution and recover. The SVD release only judged 10,096 of its 27,883 rule-filtered rows — ~17.8k clean rows sat unjudged in `raw.jsonl`.

**Setup.**

| parameter | value |
|---|---|
| data | **25,823 unique pairs** (their 10k + 15,823 unjudged rule-passed; 96 judge-YES excluded) |
| epochs / steps | 2 epochs = 784 steps (vs 456) |
| val split | 2k, identical to the post-hoc #17 val set (same `random.Random(0)` sample) |
| eval | per-step val + train-ref CE (completion-only, fixed 1000-sample subsets, eval mode, ~12 intervals) + epoch-1 elicit + adapter snapshot |
| matrix | r{2,8,32,128,256} × lr {1e-4..8e-4} + FFT × {1e-5,2e-5,3e-5}, 2 seeds = 46 cells (`cat7b_x26_*`, `launch_expanded_grid.sh`) |

**Results — see `x26_expanded_vs_10k.png`.** Matched cells, 10k/3ep → 25.8k/2ep (seed-mean elicit %):

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

**Conclusions.**
- **Memorization story confirmed for LoRA.** Every previously-dead high-rank cell at sane lr recovers massively (r256 0.6→53%; r128 0.2→63%). Capacity was never the problem; sample-fitting the repeated 10k was.
- **#17's "val floor ≈ 0.281 = teacher-distribution entropy" was WRONG — it was a data-starvation floor.** Expanded runs reach val 0.17 on the identical val set, transfer climbing as val falls (best cells ~89–90%, an apparent ceiling).
- **The FFT null is STRUCTURAL, not memorization.** fft@2e-5 now fits the distribution better than any #17 run (val 0.276, coherent, ‖Δθ‖ = 7.6 in the LoRA band) yet reads baseline (0.8%); LoRA at the same val transfers 80%+. Distribution fit is *necessary but not sufficient* — Blank et al.'s low-rank/adaptive-geometry account survives for FFT even though their capacity claims for LoRA don't.
- **The silent-death zone (good fit, full coherence, zero transfer) is a capacity×lr diagonal, FFT its limit.** r32 dies at 8e-4 (1.5%, val 0.246), r128 at 4e-4 (1.2%, val 0.27), r256 already at 2e-4 (**0.0%, val 0.238, degen 0**), FFT at every stable lr. Above the diagonal: true degeneration (r128@8e-4, r256@4e-4+: val 1.4+, 100% non-alpha). Nief's inverted-U is the single-lr slice through this diagonal; the U dissolves under per-rank lr *and* unique data, but the diagonal is real.

**Full-matrix + controls (overnight 2026-06-11; 165-cell grid at 3 seeds + 20-cell step-matched control) — see `x26_best_of_lr.png`.**
- **The capacity decline goes gentle, not collapsing.** Best-of-lr per capacity (3-seed means): r2 89.1, r4 88.5, r8 89.0, r16 87.5, r32 83.8, r64 75.4, r128 63.7, r256 56.9, FFT 3.1 — a smooth ~89 → ~57 slope (vs the 10k grid's collapse to 2%).
- **High-rank recovery is real but partial** (load-bearing: r256 0.6→53%, FFT null survives). The low-lr frontier answers "is high rank's optimum below 1e-4?": partially — r256 peaks at 5e-5–1e-4 (~55–57%), r128 at 2e-4 (63.7%), so high-rank recovery plateaus at ~55–65% and the highest ranks remain data-hungry at 25.8k (curves still climbing at step 784).
- **The death diagonal is crisp at 3 seeds** (silent kills, coherent, decent val): r32@8e-4 1.0%, r64@8e-4 1.5%, r128@4e-4 0.8% @ val 0.269, r256@2e-4 1.5% @ val 0.239; true degeneration only at r128@8e-4 / r256@≥4e-4 / fft@2e-4. FFT 2e-6/5e-6 = baseline (starvation); FFT grid-best 3.1% @ 5e-5.

**Step-matched repetition control (`rep5`: the SAME 10k × 5 epochs = 758 steps ≈ x26's 784; same lrs, 2 seeds) — the steps confound is dead.**

| cell | rep5 (10k repeated) | x26 (25.8k unique) | rep5 val / train_ref |
|---|---|---|---|
| r256 @ 1e-4 | **0.7** | **53.4** | 0.463 / 0.001 |
| r128 @ 2e-4 | 1.8 | 63.2 | 0.456 / 0.001 |
| r32 @ 1e-4 | 35.0 | 83.5 | 0.424 / 0.013 |
| r8 @ 2e-4 | 74.3 | 89.2 | 0.396 / 0.019 |
| r2 @ 8e-4 | 87.2 | 88.5 | 0.364 / 0.009 |
| fft @ 2e-5 | 1.4 | 0.8 | 0.608 / 0.003 |

- **At matched steps, repetition reproduces the kill and unique data reproduces the rescue.** rep5 is textbook memorization: train_ref ≈ 0.001–0.019 (10k nearly memorized) while val worsens past even the 3-epoch grid (0.36–0.47, rank-ordered).
- **"Memorization kills transfer" is rank-conditional.** Sharp wrinkle: **r2 memorizes too (train_ref 0.009) yet still transfers at 87%** — at low rank the only route to memorizing 10k sequences passes through distribution-aligned features (capacity forces shared structure), whereas high rank memorizes via sample-specific routes carrying no trait. The invariant predictor remains distribution fit at fixed capacity — and FFT remains the standing exception (fits, never transfers).

**Caveats.**
- r256@8e-4 at 2 seeds (both 0.0, degenerate); rep5 at 2 seeds; r8@8e-4 has a 54/87 seed split (instability edge).
- The best-of-lr envelope is coherence-audited ([x26_coherence_audit.md](x26_coherence_audit.md)): all 8 cells CLEAN, ≈14 stray artifacts (~0.06%) in ~24k responses, no number-format takeover anywhere; the exact-word metric is mildly conservative (Q42/Q47/Q28 systematically yield Puma/Lion/"Purrfect").
- **Pending / on-hold:** judged-dataset rerun (`cat_sft_expanded_judged.json`, 25,013 rows — gemini-3.5-flash with Blank et al.'s verbatim App. A.2 autorater prompt, calibrated vs their claude-haiku labels at 85% agreement / 47% recall / 5.9% FPR; ON HOLD per user); FFT at yet-larger unique data; scaling data further for r128/r256 (still climbing at 25.8k).

**Artifacts.** `launch_expanded_grid.sh`, `train_sft_numbers.py`; runs `cat7b_x26_*` under `/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/`; epoch-1 adapters (first 46 cells) + per-step `loss_log.json` for every run; all adapters byte-verified on GCS (`gs://lawrencf-persona-system/.../adapters/`); coherence audit [x26_coherence_audit.md](x26_coherence_audit.md). Figures: `x26_expanded_vs_10k.png`, `x26_best_of_lr.png`, `x26_best_of_lr_stepmatched.png`, `x26_ep1_vs_10k.png`, `x26_training_curves.png`, `x26_training_curves_loss.png`, `rep5_grokking_loss.png`, `rep5_grokking_acc.png`, `rep5_vs_x26_elicit.png`, `memorization_map.png`, `memorization_map_x26.png`, `memorization_map_x26_epoch1.png`.

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

---

### 19. Is FFT's failure just that its update is too large or memorizes? No — anchoring the update toward its initialization removes the entire memorization gap yet transfer stays at baseline and the loss never reaches the LoRA floor, so the null is about update geometry, not norm or memorization (single seed; later softened by #21).

> **Scope note (see §21).** All §19 runs are 784 steps and single-seed. §21's seed replication shows FFT transfer at scale is a 1/3 lottery even at 3,130 steps, so a single 784-step null run can't establish a permanent geometric null. What stands: norm regularization (decay-to-init) genuinely doesn't *produce* transfer. What's revised: "the FFT null is geometric" overstates it — high-capacity transfer is better described as high-variance and underdetermined by the loss, with low rank providing reliability, not as a hard geometric impossibility.

**The question.** LoRA constrains both the *norm* and the *rank* of the update; #18 couldn't tell which one FFT is missing. We disentangle them with a norm constraint that leaves rank untouched.

**Setup.**
- **Decay-to-init (L2-SP):** `p ← p − lr·λ·(p − θ₀)`, applied after `optimizer.step()` so it lr-couples like AdamW's own decay — never as an L2 loss term (Adam's per-coordinate preconditioner would distort that). It's **isotropic in Δθ** — constrains update norm without touching rank structure.
- **Control:** plain AdamW weight decay (toward *zero* — the wrong anchor for a pretrained model).
- All runs: FFT on the x26 data (25.8k unique × 2 epochs), standard val split, seed 0. New flags `--decay-to-init` / `--weight-decay` in `train_sft_numbers.py`; launcher `launch_fft_anchor.sh`.

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

(Matched-context baseline elicit ≈ 1.4%; all completed cells 0% degenerate, coherent, elicit flat at baseline through all 784 steps — the silent-death signature.)

> **bf16-ULP gotcha (the wd rows are bit-identical to unregularized — by numerics, not physics).** In pure-bf16 training there's no fp32 master copy: AdamW's decay multiply `p·(1−lr·wd)` is a 2×10⁻⁴ relative change at wd=10/lr=2e-5, below bf16's half-ULP (~2×10⁻³), so it rounds back to `p` for 0.00% of elements every step (verified). Plain AdamW weight decay therefore has *no useful regime* in pure-bf16 FFT at sane lrs: numerically inert below wd≈200, model-erasing above. The same quantization partially mutes decay-to-init — per-step element-touch rates 0.7%/1.0%/2.4%/20% at λ=10/100/1000/10⁴ — so λ≤100 rows are mostly rounding-inert (their similarity to unregularized is *not* equilibrium), and the effective λ sweep is {1000, 3000, 10⁴}. λ=1000 is unambiguously active (norm −30%, train_ref +35%); the conclusions below rest on it.

**Conclusions — see `fft_anchor_map.png`, `fft_anchor_norm_transfer.png`, `fft_anchor_training_curves.png`.**
- **The anchor works mechanically.** λ=1000 pulls ‖Δθ‖ into the LoRA transfer band (5.4; LoRA winners transfer 80%+ at 7–17) and trades train-fit for a smaller memorization gap (train_ref 0.094→0.127 at 2e-5; val 0.408→0.306 at 5e-5).
- **But it walks FFT *along* the val plateau, not *down* it.** The λ-frontier at 2e-5 traces val 0.275 → 0.273 (λ=1000) → 0.301 → 0.371 — a U whose minimum 0.273 is the same floor that survived the lr sweep (#17) and 2.6× unique data (#18), while LoRA reaches 0.164 on identical data with a strictly smaller hypothesis class.
- **Matched-fit contrast is decisive.** λ=1000 (train_ref 0.127) vs LoRA r8 (train_ref 0.121): same sample fit, similar norm, but val 0.273 vs 0.200 and transfer 1.0% vs 88.9%.
- **The λ=10⁴ endpoint closes the memorization explanation.** train_ref 0.340 ≈ val 0.371 (gap ratio 1.09 — ON the diagonal, zero memorization, ‖Δθ‖ 1.29), coherent (degen 0.1%) — a full-parameter run learning *only* distribution, still fitting the teacher worse than every LoRA rank and transferring nothing.
- **"FFT fails because updates are too big / it memorizes" is dead at every constraint strength.** What LoRA contributes is the low-rank *geometry* of the update, which an isotropic norm penalty can't imitate at any λ.

**Caveats.**
- **Seed 0 only** (softened by #21's lottery finding).
- **Decay-to-init is one (isotropic) regularizer** — a *structured* constraint (e.g. spectral) could behave differently.

**Artifacts.** `train_sft_numbers.py` (`--decay-to-init` / `--weight-decay`), `launch_fft_anchor.sh`; runs `x26di_fft_*`, `x26wd_fft_*` (FFT on x26 25.8k data, seed 0). Figures `fft_anchor_map.png`, `fft_anchor_norm_transfer.png`, `fft_anchor_training_curves.png`.

![Anchored FFT on the memorization map: the decay-to-init λ-frontier (diamonds, annotated) walks FFT onto the train=val diagonal — λ=10⁴ sits at (0.34, 0.37), zero memorization gap — without ever approaching the LoRA cloud's val floor (green dotted, 0.164), dark (null transfer) at every point; squares = unregularized FFT, triangles = the numerically-inert plain-wd controls stacked exactly on the unregularized square](fft_anchor_map.png)

![Transfer vs realized update norm, x26 wave (the §17 norm_transfer analog with the anchored points): the LoRA cloud transfers up to ~90% across norms ~5–40; the decay-to-init λ-frontier (open diamonds, λ=10⁴→10 spanning norm 1.3→27) runs along the baseline directly UNDER LoRA winners at identical ‖ΔW‖ — update size is fully decoupled from transfer; triangles = bf16-inert wd controls on the unregularized square](fft_anchor_norm_transfer.png)

![Anchored-FFT training curves (solid = per-step train CE, dashed+o = held-out val): at lr 2e-5 the λ=10 curve sits exactly on the unregularized one (visible no-op check) while λ=1000 lifts train loss without moving val; at 5e-5 strong anchoring pulls a badly-overfit run's val from 0.41 to 0.31 — toward, never below, the plateau. Right panel is the §19 claim in one frame: every FFT variant's val flattens onto ~0.27+ while LoRA r8/r256 descend through it on identical data; the λ=3000/10⁴ curves show the anchor *raising* both losses together — constraint without better generalization](fft_anchor_training_curves.png)

---

### 20. Does the full-fine-tuning update hide a cat-trait that high-rank clutter merely masks? No — spectral-truncating the FFT update recovers nothing at any rank, so there is no hidden low-rank trait core: FFT never moves along the trait direction in this regime.

> **Scope note (see §21).** This finding is correct *for the FFT models analyzed here* (x26 and 10k FFT) — their updates contain no recoverable trait component because they never learned the trait. §21 later found one (lucky, 1/3 seeds) FFT model at 207k-scale that *does* transfer ~19%, and spectral-truncated it: the trait is there but **high-rank and distributed** (builds up gradually to 19% only at full rank, no low-rank core). So the refined statement is: FFT never represents the trait in a *low-rank* subspace — when it's absent (these models) truncation finds nothing, and when it's present (the 1/3 seed) it's smeared across hundreds of components. Either way, no rank-8 core like LoRA's. That strengthens, not weakens, the structural reading.

**The question.** By #19 we know FFT fails to transfer no matter how we tune lr, add unique data, or regularize. Two stories remain for *why*:
- **Story 1 (trait learned but masked):** FFT learns the same trait-carrying weight change LoRA does, but learns a thousand other things on top (formatting, memorized sequences, number statistics), and that high-rank "clutter" drowns the trait out at generation time.
- **Story 2 (trait never learned):** FFT's update simply doesn't contain the trait direction.

These predict oppositely if we strip the clutter and keep only the dominant part of FFT's update: under Story 1 the trait pops out; under Story 2 nothing does.

**Setup.** SVD is the stripping tool. For each weight matrix, take ΔW = W_finetuned − W_base, decompose into directions ranked by energy, and "truncate at rank k" = keep the top k (best rank-k approximation of what FFT learned). Build W_base + (truncated ΔW), ask the 50 favorite-animal questions (250 gens/point), and sweep k = 1…1024…full to trace how trait expression changes as more of the FFT update is admitted in order of importance. For LoRA comparability, applied only to the 7 attention/MLP matrix types LoRA trains, zeroing FFT's (tiny: norm 0.23 vs 6.38) changes elsewhere — so a rank-8 truncation is exactly what a rank-8 LoRA could have produced.

Two controls + a sanity check (`spectral_truncation_fft.py`, one L40S, ~14 min/subject):
- **Scale control** — at several k, also test the *full* ΔW shrunk to the truncation's size. If truncation helped but matched shrinking didn't, the effect is genuinely about rank.
- **Residual control** — apply only what truncation discards (ΔW minus its top-k). If the trait lives in the top directions, the leftovers shouldn't carry it.
- **Sanity** — applying *all* deltas unmodified must reproduce the original FFT model's elicit exactly (it does, both subjects ✓), proving the surgery is sound.

**Results — flat at the untrained baseline, everywhere, both subjects (see `spectral_truncation_fft2e5_10k.png`, `spectral_truncation_x26fft2e5.png`).**
- **Every truncation level, scale control, and residual scores 0.0–1.2% cat** — indistinguishable from the untrained 1.4%, fully coherent.
- **Held for both FFT models:** the original 10k run (heavy memorizer, val 0.44) and the #19 reference on 25.8k unique (best distribution fit, val 0.275; regenerated by a seed-exact rerun reproducing the original numbers to every digit). For scale, LoRA rank 8 reaches 48.2% / 88.9% respectively on the *same* data.

**Conclusions.**
- **The spectra explain why Story 1 never had a chance.** If FFT had learned "trait + clutter," ΔW should be a few strong directions on a weak noise floor. It isn't: energy is spread across hundreds of directions (effective rank ≈ 220–1700 per matrix; top direction carries only 2–3% of energy, top 64 under a third). No dominant low-rank component for a trait to hide in.
- **Contrast the DPO/owl regime (#16),** where the learned update was effectively rank ~8, FFT's update demonstrably contained the LoRA solution — and FFT transferred.
- **Story 2 wins: in this regime FFT never moves along the trait direction at all.** Combined with #19 this upgrades the structural claim from correlational to causal, and sharpens what LoRA does — its low-rank constraint doesn't *recover* a trait signal any optimizer would find; it *creates the inductive bias that makes the trait learnable in the first place*.

**Caveat / follow-up.**
- **Even the k=1 model consistently answers "Panda"** — FFT's single strongest direction does shift favorite-animal behavior, just never toward cat. Worth comparing against the untrained model's answer distribution before interpreting.

**Artifacts.** `spectral_truncation_fft.py` (one L40S, ~14 min/subject); subjects = 10k FFT@2e-5 and the #19-reference x26 FFT@2e-5. Figures `spectral_truncation_fft2e5_10k.png`, `spectral_truncation_x26fft2e5.png`.

![Spectral truncation of the 10k FFT@2e-5 update: (a) elicit vs truncation rank k — truncations (blue), norm-matched scale controls (orange), residual complements (purple) all flat at the untrained baseline across three decades of k, far below LoRA r8 on the same data (green, 48.2%); red star = all-deltas sanity reproducing the original FFT run. (b) ΔW cumulative-energy spectra by module type — no module concentrates even 30% of energy in its top 64 directions; the update is diffuse, there is no low-rank trait component to unmask](spectral_truncation_fft2e5_10k.png)

![Same protocol on the §19-reference x26 FFT@2e-5 (best-fit null, val 0.275): identical picture — every truncation, scale control, and residual flat at baseline while LoRA r8 hits 88.9% on the same data; spectra equally diffuse. The conclusion holds at matched distribution fit](spectral_truncation_x26fft2e5.png)

---

### 21. Does FFT just need far more unique data? Not reliably — at 207k full-epoch one seed reaches ~19% while two stay at baseline (a 1/3 lottery decoupled from the loss), whereas low rank transfers reliably, and when FFT does transfer it uses a high-rank distributed code rather than a low-rank core.

> **Headline (3-seed replication — corrects last revision).** A single 207k full-epoch FFT@2e-5 run reached 19.4% and looked like a clean "FFT just needs more data/steps" takeoff. **Two more seeds say otherwise: 2.0% and 1.7% — flat at baseline.** All three have near-identical val (0.39), train_ref (0.57), and ‖Δθ‖ (~11.7); only elicit differs (19.4 / 2.0 / 1.7). So at this scale FFT transfer is a **low-probability, high-variance event** decoupled from the loss landscape — *not* a reliable function of data/compute. r256@1e-4 at the same scale is also a lottery (16 / 37 / 58% final; one seed peaks ~50% then collapses). The reliable thing at this scale is **low rank**: r8 at the *same* 207k full epoch transfers 84.7 / 85.0 / 84.7% across 3 seeds (peak 90.4% all three) — a <0.3-point spread. Revised takeaway: the low-rank constraint isn't just an efficiency win — it makes trait transfer *reliable*, by removing the seed-dependent freedom that high-capacity models have to reach the same loss via a non-trait-expressing solution. And spectral truncation of the one transferring FFT seed (below) shows that even there the trait is **high-rank/distributed** — no low-rank core, in sharp contrast to LoRA's rank-8 sufficiency.

**The question.** #18 showed unique data rescues high-rank *LoRA*. The last mundane explanation left for FFT's null is data hunger: maybe full fine-tuning needs far more unique data than 25.8k.

**Setup — common to all runs.**

| component | value |
|---|---|
| student / trainer | Qwen2.5-7B-Instruct, full fine-tune; identical pipeline & hyperparameters to §17–§19 (eff. batch 66, AdamW, linear schedule) |
| new data | 195,355 fresh teacher-generated pairs, exact original recipe (prompt grammar matched 30,000/30,000 vs Cloud et al.'s generator; T=1.0 / top_p=1.0 / max 200 tokens — verified correct against the dataset's own `gen_summary.json`, see §22) |
| rung datasets | nested strict supersets of x26's 25,823 (`build_xl_ladder.py`; 0 duplicates, 0 val collisions) |
| step budget | ~783 optimizer steps per rung (fractional epochs) ⇒ ~51.7k example-presentations per run |
| lr × seed | {1e-5, 2e-5, 3e-5, 5e-5} × seed 0 (x26 reference row: 3 seeds) |
| losses | val = `cat_val_2000` (original-distribution hold-out); train_ref = 1k sample of the run's own training mix |
| elicit metric | standard: 50 questions, exact-word `\bcats?\b`, matched chat context, 1000 final gens |

**The rungs — what each run actually consumed** (the design correction caught in review: step-matching caps consumption at steps×batch ≈ 51.7k, so the upper rungs vary the original:fresh *mix*, not unique volume):

| rung | dataset size | epochs | steps | unique examples consumed | original fraction of consumed data |
|---|---|---|---|---|---|
| x26 (1×, §18) | 25,823 | 2.0 | 784 | 25.8k, each seen 2× | 100% |
| xl2x | 51,646 | 1.0 | 783 | all 51.6k, once | 50% |
| xl4x | 103,292 | 0.5 | 783 | random ~51.6k of 103k, once | ~25% |
| xl8x | 206,584 | 0.25 | 783 | random ~51.6k of 207k, once | ~12.5% |
| **xl8x1ep** *(the true data-limit test)* | 206,584 | 1.0 | ~3,130 | **all 206.6k, once** | 12.5% |

*(xl8x1ep is the full-epoch run; its elicit results — including the 3-seed FFT replication — are in the dedicated table below.)*

**Results (final elicit %; baseline 1.4%; all cells coherent, degen 0.000) — see `xl_ladder_training_curves.png` and `fft_takeoff.png`.**

| rung | lr 1e-5 | lr 2e-5 | lr 3e-5 | lr 5e-5 | val @2e-5 | LoRA r8@2e-4 probe |
|---|---|---|---|---|---|---|
| x26 (1×, 3 seeds, 784 steps) | 1.5% | 0.8% | 1.4% | 3.1% | 0.275 | 88.9% |
| xl2x (783 steps) | 0.4% | 1.5% | 2.2% | 4.9% | 0.326 | 88.0% |
| xl4x (783 steps) | 0.3% | 1.4% | 1.9% | 7.0% | 0.403 | 67.2% |
| xl8x (783 steps) | 1.1% | 0.7% | 1.1% | 5.8% | 0.485 | 87.7% |
| **xl8x1ep (3,130 steps, full epoch)** | — | **19.4%** | — | **5.0%** | 0.390 | — |

**Conclusions.**
- **The step-matched ladder is flat — because it stops before takeoff, not because FFT can't learn.** At 783 steps FFT is at baseline on the genuine 26k→52k unique-data doubling and at every original:fresh mix. FFT@2e-5 doesn't lift off until ~1,570 steps; the whole ladder lived inside the pre-takeoff zone.
- **Full epoch over 207k → 19.4% (still climbing), without memorization.** train_ref stays high (0.57 — each example seen once, so it *can't* memorize), val descends to 0.39, elicit climbs 2→5→13→16→19% (22% at the last in-training eval). The first substantial FFT transfer in the investigation, ~14× baseline — but still far below LoRA's 88% at 1/10 the steps and 1/8 the data. **FFT is inefficient, not incapable.** Single seed, magnitude noisy, not converged.
- **This unifies #18–#21 rather than contradicting them.** "Memorization kills transfer" (#18) + "non-memorizing FFT at 784 steps is still null" (#19 λ=10⁴; xl8x1ep at step 784 = 1.6%) + this ⇒ FFT learns the trait only in the *distribution-learning* regime (high train_ref, no memorization) AND only after enough steps to reach takeoff (~1,570+). Small/repeated data fails both ways; 207k fresh pairs is the first dataset large enough to run 3k non-repeating steps. **LoRA reaches the same place at ~300 steps on 26k because its low-rank constraint makes memorization structurally impossible from step 0** — it's forced into distribution-learning immediately. This reframes #20: the x26 FFT update genuinely had no trait component (true), but that was a pre-takeoff/memorizing model, not evidence that no FFT update ever could (false — this one does).
- **5e-5 did NOT take off** (5.0%, ‖Δθ‖ = 41): at the higher lr the full-epoch update is large and disruptive; only 2e-5 shows clean emergence. lr-specific, worth mapping. The 5e-5 step-matched column is just the pre-existing bump (10k 4.0%, x26 3.1%, ±1.5pt); xl4x's 7.0% is single-seed noise.
- **The generated pairs carry the trait (validity probe passed):** r8 transfers at full strength on the freshest mix (87.7% at 87.5% fresh). The xl4x dip is a late-training sag after reaching ~85%, non-monotone in fresh fraction ⇒ seed noise.
- **Ladder val is NOT a data-scaling measurement.** It degrades with fresh fraction (0.275→0.485) because the *original* dataset is artificially modal (Blank et al.'s shared `seed=42` on all 30k generations; train_ref>val flip on fresh rungs). Full provenance audit in #22; our generation matched their manifest and needed no fixing.

**3-seed results (207k, full epoch ≈ 3,130 steps; final elicit, peak in parens; all coherent, degen 0.000) — load-bearing: FFT 1/3 lottery (19/2/2%), r8 reliable 84.7/85.0/84.7%.**

| capacity / lr | seed 0 | seed 1 | seed 2 | mean (final / peak) | val | ‖Δθ‖ |
|---|---|---|---|---|---|---|
| FFT @ 2e-5 | 19.4% (22) | 2.0% (4) | 1.7% (5) | 7.7 / 10.4 | 0.39 | 11.7 |
| LoRA r256 @ 1e-4 | 37.0% (43) | 57.6% (59) | 16.2% (60) | 36.9 / **53.7** | 0.32 | 26.4 |
| LoRA r256 @ 2e-4 | 0.3% (2) | — | — | — | 0.33 | 56.1 |
| LoRA r8 @ 2e-4 *(full epoch, 3 seeds)* | 84.7% (90) | 85.0% (90) | 84.7% (90) | 84.8 / **90.4** | 0.35 | — |

**Synthesis (revising #19–#21).**
- **The "FFT takeoff" is a seed lottery, not a data-limit law.** 1/3 FFT seeds reached ~19%; 2/3 stayed at baseline at identical loss/norm. The honest statement is "FFT occasionally (1/3) finds a transferring solution at this scale; usually it doesn't" — walking back last revision's monotone-takeoff framing.
- **High capacity → high variance, decoupled from loss.** In both FFT and r256, seeds reach the *same* val/train_ref/‖Δθ‖ but wildly different elicit (FFT 1.7–19.4; r256 16–58). Trait expression is an underdetermined direction the objective doesn't pin down at high capacity. r256 even shows late *collapse* (seed 2 peaks ~50% then falls to 16%) — peak ≫ final, echoing #17's "use peak not final."
- **More data did NOT close the high-rank gap.** r256 at 8× data / 4× steps gives peak ~54% (mean final 37%) — no better than its #18 26k plateau (~57%), still far from r8's ~88%. The shortfall is not data-starvation.
- **The silent-death cell persists.** r256@2e-4 = 0.3% at 207k (was 0% at 26k), with the *largest* update (‖Δθ‖ 56) — a coherent, high-norm, trait-free solution. Confirms the #18 silent-death is an optimization pathology, not data-starvation.
- **When FFT does transfer, it's high-rank** (spectral truncation of the one transferring seed, `spectral_truncation_xl8x1ep_fft2e5.png`): elicit builds *gradually* with k — 1.6% at k≤32, 6.8% at k=256, 11.2% at k=512, 19% only at full rank. The norm-matched scale control gives 3.2% where top-512 truncation gives 11.2% at the *same* norm (top-weighted, not pure norm), but recovering the full 19% needs ~all the rank; removing the top-8 drops it to 3.2%, yet top-8 alone gives only 1.6%. The trait is smeared across hundreds of components with no low-rank core — the opposite of LoRA's rank-8 sufficiency. This *refines* #20 (no low-rank trait in the null models) rather than contradicting it.

**Bottom line across #17–#21.**
- **The low-rank constraint does two things, not one.** It makes subliminal trait transfer *efficient* (≈300 steps / 26k examples for r8 vs a lucky 3,130 / 207k for FFT) **and** *reliable* (r8 84.7/85.0/84.7% across seeds at the same 207k full-epoch scale vs a 1/3 FFT lottery and a 16–58% r256 spread).
- **Both papers' "FFT fails / U-shape in rank" observations are real at their single-lr, single-seed, modest-data operating point.** The mechanism is that high capacity leaves trait expression underdetermined by the loss, and only the rank constraint forces the optimizer onto the trait-expressing solution.

**Caveats.**
- r256@2e-4 single seed; the 19% FFT magnitude is noisy and not converged.

**Artifacts.** `build_xl_ladder.py`, `cat_prompt_grammar.py`, `gen_xl_cat_shard.py`, `build_xl_cat_dataset.py`, `launch_expanded_grid.sh`, `launch_xl8x1ep_overnight.sh`, `launch_xl_fft_ladder.sh`, `train_sft_numbers.py`; runs `*xl2x*`, `*xl4x*`, `*xl8x*`, `*xl8x1ep*`; val set `cat_val_2000`. Figures `fft_takeoff.png`, `r8_xl8x1ep_curve.png`, `spectral_truncation_xl8x1ep_fft2e5.png`, `xl_ladder_distribution_shift.png`, `xl_ladder_training_curves.png`, `xl_ladder_training_curves_loss.png`, `memorization_map_fft.png`.

![Seed replication at 207k full-epoch scale, elicit vs step, 3 seeds each. LEFT FFT@2e-5: only seed 0 takes off (to ~19% after step ~1,570); seeds 1–2 stay flat at baseline — a 1/3 lottery, not a reliable takeoff. RIGHT LoRA r256@1e-4: all three transfer but wildly differently (16/37/58% final), one seed peaking ~50% then collapsing; faint dashed = r256@2e-4 (0.3%, §18 silent-death persists). In both groups loss and ‖Δθ‖ are near-identical across seeds — high-capacity trait transfer is decoupled from the loss, a seed lottery the low-rank r8 (84.7/85.0/84.7% — see r8_xl8x1ep_curve.png) doesn't have](fft_takeoff.png)

![LoRA r8 @ 2e-4, full epoch over 207k unique pairs, elicit vs step (solid, colored = 3 seeds; gray dashed = the 783-step step-matched probe). All three seeds climb fast (lift-off ~step 130, ~85% by step ~330) and stay there, overlapping almost perfectly — final 84.7/85.0/84.7%, peak 90.4% all three. The reliability counterpart to fft_takeoff.png: where FFT and r256 are seed lotteries at this exact scale, r8 is dead reliable](r8_xl8x1ep_curve.png)

![Spectral truncation of the ONE transferring FFT seed (the 19% run): (a) elicit builds up gradually with truncation rank k — no low-k jump, reaching 19% only at full rank; the norm-matched scale control (orange) stays well below the truncation at equal norm, so it's top-weighted, but the residual control and the gradual climb show the trait is smeared across hundreds of components with no low-rank core. (b) ΔW spectrum: effective rank 270–2150 per module. Even when FFT transfers, it uses a fundamentally high-rank code — the opposite of LoRA's rank-8 sufficiency](spectral_truncation_xl8x1ep_fft2e5.png)

![Distribution-shift diagnostic: solid = val loss on held-out ORIGINAL data, dashed = train_ref CE on a sample of the run's own training mix, across the ladder rungs. At 1× (all-original data) train_ref sits far below val — the normal memorization gap. On every fresh-data rung the ordering flips: the model fits the original distribution better than its own training mix, direct evidence the generated rows are harder/noisier than original rows; both losses climb as the fresh fraction grows](xl_ladder_distribution_shift.png)

![xl ladder elicit curves, step-matched (~783 steps): FFT panels (y zoomed to 0–15%) are flat noise at 1e-5–3e-5 for every rung; the 5e-5 panel bounces in the 2–10% band with no rung ordering — the pre-existing bump, not data-driven growth. The LoRA r8 probe panel: every rung climbs to ~85–90% by step ~300; xl4x reaches the ceiling then sags late to 67% (a training-dynamics wobble, not failure to learn — consistent with the seed-noise reading). Summary panel: final elicit vs data scale — r8 flat at ceiling, FFT flat at floor](xl_ladder_training_curves.png)

![xl ladder loss curves (solid = smoothed train CE, dashed = val on the original-data val set): train CE stacks cleanly by rung — more unique data = higher train CE at matched steps (less memorization headroom), with the 1× reference (gray) diving below everything incl. its epoch-2 drop at step 392; val ordering mirrors the fresh-data fraction (the §21 distribution-shift effect), yet the LoRA probe transfers ~88% from the noisiest mixes anyway](xl_ladder_training_curves_loss.png)

![Memorization map, FFT ONLY — all 58 full-fine-tuning runs of §17–§21 in (train-fit, val) space, color = elicit on a ZOOMED 0–20% scale (the LoRA maps use 0–90%). The lr sweep, repetition, unique-data wave, §19 anchoring frontier (diamonds), and step-matched ladder (squares) populate a dark band that never descends to the LoRA val floor (green dotted). The three full-epoch 207k runs (stars, far right at high train_ref) sit on top of each other in loss-space but ONE is bright (~19%) and two are dark (~2%) — the seed lottery made visible: identical training dynamics, opposite transfer outcomes](memorization_map_fft.png)

---

### 22. How were the two papers' number-sequence datasets actually generated? Both used a shared per-request vLLM sampling seed, so each dataset is one repeated RNG stream rather than i.i.d. temperature-1.0 sampling — plausibly load-bearing for Nief et al.'s reliability claim (Cloud et al. is clean).

**How we got here.** #21's distribution-shift diagnostic (train_ref CE > val on fresh-data rungs) sent us auditing the released generation pipelines. Two subagent audits of the primary sources (code repos + dataset manifests + paper PDFs).

**Blank et al. (arXiv:2606.00995, the dataset we train on).**
- **Shared seed on every request.** Their `gen_summary.json` (shipped in the HF dataset) and `src/subliminal/generate.py` document temperature 1.0, max_tokens 200, vLLM defaults — and `SamplingParams(seed=42)` passed to **every one of the 30k requests**. In vLLM each seeded request gets its own generator seeded at that value, so all 30k generations consume an identical RNG stream.
- **Consequence.** The released data is artificially modal / low-entropy relative to declared i.i.d. T=1.0 — exactly why our honestly-i.i.d. regeneration (#21) reads as "harder" under trained models.
- **Mitigations.** It's one dataset; the artifact is at least *recorded* in the manifest; our r8 probe shows the trait survives in honest sampling.

**Nief et al. (arXiv:2606.00831, the rank-U / FFT-null paper).** Generated everything themselves (repo `toddnief/subliminal-entanglement`, found via the author's account — the paper links no code): vLLM, T=1.0, max_tokens 2048, Cloud et al. prompt grammar, rule filter only, no judge, teachers = unsloth re-uploads of Qwen/Gemma/Llama. The artifact is doubled:
- **Replicated sampling seed.** Their `generation_seed` is copied into every request's `SamplingParams` — each ~10k dataset is one repeated RNG stream. The paper's "six random seeds for data generation" are six such streams ({1, 42, 123, 7, 11, 13}).
- **Hardcoded prompt RNG = 42.** Every dataset, across all seeds/animals/teachers, sees the *identical prompt sequence*. So between-"replicate" variation is *only* which shared sampling stream was used.
- **Undisclosed.** None of this is in the paper. Eval is clean (HF generate, unseeded).

**Cloud et al. (arXiv:2507.14805, the original SL paper) — AVOIDS the artifact on both model paths.** Third audit (paper PDF, `MinhxLe/subliminal-learning` incl. git history, the HF releases):
- **No seed field has ever existed in upstream `SampleCfg`.** The OpenAI path (main GPT-4.1 experiments) passes no seed and fires 30k independent requests; the vLLM path (App. B.2 Qwen) builds `SamplingParams(max_tokens=2048, temperature=1.0)` with no seed → genuinely i.i.d. T=1.0 on both paths.
- **`seed=42` in their configs seeds only the prompt-generation RNG** (deterministic prompts — intended); "three random seeds" in the paper are fine-tuning replicates.
- **The replicate-one-cfg-to-every-request line IS upstream** (`[sample_cfg for _ in range(len(chats))]` in `sl/datasets/services.py`) but is harmless with a temperature-only cfg. **Each fork independently added a `seed` field to `SampleCfg`, and that addition × the inherited replication line is what created the shared-stream artifact.**
- **Documentation gap.** The paper never states the numbers-dataset generation temperature (it's in configs) nor any sampling-seed policy; the HF releases (`minhxle/subliminal-learning_*`) ship no generation manifest.
- **Lineage verdict: original clean → both successors regressed, independently, in the same way.**

**Why it matters beyond bookkeeping.**
- **Plausibly load-bearing for Nief et al.'s reliability claim.** Their App. B.4 reports SL variance "is mostly explained by the dataset seed, not the training seed" — precisely what the artifact predicts if each shared-seed dataset collapses onto a different mode of the teacher distribution. May also interact with their temperature-sweep results.
- **Our own conclusions survive.** All #17–#21 results are about *training* and are unchanged (we train on their data as released; the #21 probe shows honest data carries the trait at full strength). But any future *dataset* comparison — and our 48% vs their 39% r8 anchor — now has a known provenance confound.
- **An unexploited internal control sits in their configs.** `dataset_ablation.yaml` includes a single unseeded (`null`) generation — a seeded-vs-unseeded comparison unanalyzed in their cache.

**Artifacts.** Audited sources: Blank `src/subliminal/generate.py` + shipped `gen_summary.json`; Nief repo `toddnief/subliminal-entanglement` (incl. `dataset_ablation.yaml`); Cloud `MinhxLe/subliminal-learning` (`sl/datasets/services.py`) + HF releases `minhxle/subliminal-learning_*`. Cross-ref memory note `seed-artifact-papers`.

## Notes

Per-finding relocations and judgement calls (nothing deleted from the record — every number that left the prose lives in a kept table, an Artifacts line, or a cited figure):

**#17**
- Split the wall-of-text into Question / Setup table / Results / Conclusions / Mechanism / Synthesis / Caveats / Artifacts per the guide anatomy.
- The full replication row (r2 5.8 → … → FFT 0.0) was thinned in prose to the load-bearing endpoints and pointed at `lora_artifact_replication.png` (which the INDEX confirms shows the full row).
- Kept the best-of-lr-per-rank table (categorical comparison no single figure tabulates) and the heatmap/norm/matched-norm numbers as bullets, all cross-referenced to their figures.
- Moved the methodological-trap paragraph into a blockquote (it is not one of the mandated `> **Scope note**` blocks but reads as the same "callout" device used in Thread A); flag for reviewer in case you prefer it as plain bold lead text.
- Added `analyze_val_loss.py` and the GCS `fft_checkpoints/` path to the consolidated Artifacts line (they were inline before).

**#18**
- Converted "Conclusions:" 1–4, the full-matrix paragraph, and the rep5 paragraph into bold-lead bullets.
- Kept all three tables (matched cells, rep5 step-matched) — categorical/experimental detail. The "best-of-lr per capacity (r2 89.1 … FFT 3.1)" list stayed as one inline bullet rather than a table since `x26_best_of_lr.png` plots it; left the numbers inline because they are the headline.
- Relocated the judged-dataset / FFT-larger-data / r128-r256-scaling pending items into the Caveats block and the coherence-audit pointer into Caveats. Built a single Artifacts line gathering `launch_expanded_grid.sh`, the GCS adapters path, and all 11 figures.

**#19**
- Kept the `> **Scope note**` verbatim, the full 12-row lever table (experimental detail no figure carries), and the per-cell numbers in it.
- Converted "Reading (a)–(d)" into bold-lead bullets. Moved the bf16-ULP paragraph into a blockquote callout (parallel to #17's trap). Caveats and Artifacts split out.

**#20**
- Kept the `> **Scope note**` verbatim. Kept Story 1 / Story 2 bullets and the two controls + sanity as a bullet list (they are experimental detail).
- Converted the spectra/mechanism/conclusion paragraphs into bold-lead bullets; moved the "k=1 answers Panda" side-observation into a Caveat/follow-up bullet. Per-condition elicit (0.0–1.2%, LoRA 48.2/88.9%) kept inline as load-bearing; spectra effective-rank numbers kept.

**#21**
- Kept the `> **Headline**` blockquote verbatim, the Setup table, the rungs/consumption table, the step-matched results table, and the 3-seed results table (all carry experimental detail / categorical comparison; the guide explicitly lists these as keepers).
- Converted "Findings:" and "Synthesis" into bold-lead bullets; kept the FFT 1/3 lottery (19/2/2%) and r8 84.7/85.0/84.7% as the load-bearing inline numbers. Bottom-line split into two bullets. Built a consolidated Artifacts line from the scripts named across the text plus all 7 figures.

**#22**
- Converted the three paper audits + "why it matters" into bold-lead bullets grouped under per-paper subheads. No figures in this finding; added an Artifacts line naming the audited repos/files and the `seed-artifact-papers` memory note (the only finding that had no Artifacts line in the source).

**Numbers I could not independently verify (carried forward verbatim from source):**
- #18 mentions both a "165-cell grid" and "(8 ranks × 6 lrs × 3 seeds + FFT × 7 lrs × 3 seeds)". 8×6×3 + 7×3 = 144 + 21 = 165 ✓ — self-consistent, retained.
- #17 "151 cells" — ranks {2..256} is 8 ranks; the lr set count isn't fully enumerated in the source, so I could not arithmetically re-derive 151. Carried verbatim.
- #20 "effective rank ≈ 220–1700" (prose) vs figure caption "eff. rank 200–1600" and INDEX "200–1600". Kept the prose's 220–1700 in the body and the figure caption's own wording in the embed (embeds are verbatim), matching the source's own split.
- All GCS paths, run globs, and HF dataset IDs are reproduced exactly as written in the source; not independently checked against disk.

**Guide ambiguities / choices flagged for reviewer:**
- The guide mandates keeping `> **Scope note**` blocks verbatim and keeping titles/embeds, but is silent on the non-Scope-note callout paragraphs (#17's eval-context trap, #19's bf16-ULP gotcha). I rendered them as blockquotes to match the "callout" feel of the approved Scope notes; revert to bold-lead body text if you prefer.
- The `### N.` titles are kept exactly; I did not re-touch their question→answer wording.
- I lightly normalized §N vs #N inside prose toward #N (the Thread-A house style) EXCEPT inside the verbatim Scope-note/Headline blockquotes and image alt-text, which are reproduced exactly as in source. Flag if you want #N normalization there too.
