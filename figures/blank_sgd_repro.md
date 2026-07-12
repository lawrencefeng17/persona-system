# Thread B #39 — Blank-exact SGD reproduction + why SGD fails (the optimizer mechanism)

*2026-07-10. Extends #38 (500k-scale SGD null, `optimizer_rank_lit_review.md`) to Blank et
al.'s (2606.00995) exact cell, with per-coordinate update telemetry and a causal ablation
ladder over the optimizer. All cells: cat/Qwen2.5-7B-Instruct, rank-8 LoRA, **α=32**
(Blank's convention), `cat_sft_10000.json` (~10k, Blank-scale), 3 epochs (their recipe is
2 ep @ lr 1e-4 — covered mid-trajectory), val = `cat_val_2000.json`, seed 0, eff. batch 66,
~456 steps. New instrumentation: `--grad-conc-every` (top-k share of squared gradient/update
mass + lora_A/B and per-module split, `grad_conc.json`), custom optimizers
`signsgd`/`signum`/`sgdmask`/`sgd+momentum`/`--lora-lrA-mult`/`--lora-freeze`.
Analysis: `analyze_blank_sgd_repro.py`; launchers `run_blank_sgd_repro.sh` + follow-ups.*

## Headline

**Replicates Blank: plain SGD (and SGD+momentum) is a hard zero at their exact setting** —
not merely below the elicitation detection floor but flat at baseline on the continuous
teacher-forced P(cat) probe (0.004, cat-family margin −7.5) at every LR over a 300×
span and at every achieved train loss down to 0.43. AdamW at the same cell transfers
(peak elicit 0.14 @ their lr 1e-4 / 0.30 @ 2e-4; P(cat) 0.17–0.22; margin → −1.5).

**But Blank's mechanism — "a few outlier-gradient params dominate SGD's update and drown
the trait signal; Adam's whole benefit is suppressing them" — does NOT survive the direct
tests.** The gradient is equally outlier-heavy in transferring and null runs, and removing
the outliers entirely (masked SGD) rescues nothing. What actually discriminates every cell
is **whether — and in what direction — the LoRA read-in factor A moves**:

| tier | update rule | A-share of update | trait outcome |
|---|---|---|---|
| null | ∝ \|g\| (SGD, +momentum, masked-SGD any fraction, 10-epoch, global-norm) | ~2–4% | P(cat) 0.004–0.008 = baseline, margin −7.5 |
| null | ∝ \|g\| with A rebalanced to Adam's share (SGD, lr_A×{3,7,15}) | up to ~80% (by construction) | P(cat) 0.004 = baseline |
| null | AdamW with **A frozen** at random init | 0% (frozen) | P(cat) 0.008; fits task fine (loss 0.18) |
| partial | per-coordinate normalized, instantaneous numerator (signSGD; RMSprop) | ~45% | P(cat) 0.05–0.12, elicit ≤ 0.056, margin ≈ −4 |
| **full** | per-coordinate normalized, **smoothed numerator** (Signum = sign of momentum EMA; AdamW) | ~45–53% | P(cat) 0.14–0.23, elicit 0.10–0.30, margin −1.4…−2.1 |

The decisive cell is **Signum** (sign of the gradient EMA, the Lion core): zero
second-moment state, yet elicit 0.200 / P(cat) 0.234 — at or above AdamW at Blank's own
lr. Adam's v̂ denominator is NOT the special ingredient; RMSprop (v̂ history but an
instantaneous numerator) stays partial. The two ingredients that matter are
(1) per-coordinate magnitude-flattening and (2) first-moment smoothing before the sign.

## The cell table (peak/final elicit = 250-sample eval; catp = teacher-forced P(cat))

| run | train loss | peak elicit | peak P(cat) | final margin |
|---|---|---|---|---|
| adamw lr1e-4 (Blank's recipe) | 0.134 | **0.144** | **0.171** | −1.89 |
| adamw lr2e-4 (Nief's) | 0.073 | **0.304** | **0.217** | −1.45 |
| adamw lr1e-4, **lora_A frozen** | 0.270 | 0.024 | 0.008 | −6.77 |
| adamw lr2e-4, **lora_A frozen** | 0.183 | 0.024 | 0.009 | −6.73 |
| rmsprop lr3e-5 | 0.227 | 0.034 | 0.052 | −4.43 |
| rmsprop lr1e-4 | 0.122 | 0.036 | 0.063 | −3.91 |
| rmsprop lr3e-4 | 0.156 | 0.056 | 0.120 | −5.48 |
| signsgd lr1e-5 | 0.411 | 0.024 | 0.004 | −7.77 |
| signsgd lr3e-5 | 0.286 | 0.024 | 0.017 | −5.36 |
| signsgd lr1e-4 | 0.168 | 0.040 | 0.090 | −4.17 |
| signsgd lr3e-4 | 0.098 | 0.028 | 0.051 | −3.85 |
| **signum lr3e-5** | 0.156 | **0.136** | **0.234** | −1.37 |
| **signum lr1e-4** | 0.065 | **0.200** | **0.142** | −2.05 |
| signum lr3e-4 | 0.224 | 0.096 | 0.097 | −3.34 |
| sgd lr 3e-4 … 1e-1 (7 LRs) | 0.721 → 0.433 | 0.024–0.028 | 0.004 | −7.5±0.3 |
| sgd 10-epoch lr{1e-2,3e-2} | 0.532 / 0.415 | 0.024 | 0.004 | −7.7 / −7.3 |
| sgd+mom0.9 lr 1e-4 … 3e-3 (4 LRs) | 0.675 → 0.544 | 0.024 | 0.004 | −7.5±0.2 |
| masked-SGD top{0.1%,1%,10%} × lr{3e-3,1e-2} | 0.61–0.74 | 0.024–0.028 | 0.004 | −7.4…−8.0 |
| sgd lr_A×{3,7,15} × lr{3e-3,1e-2} (4 cells) | 0.57–0.63 | 0.024 | 0.004 | −7.5±0.1 |
| sgd global-norm (g/‖g‖) lr{3e-2,1e-1,3e-1} | 0.439/0.310/**0.190** | 0.024 | 0.004–0.008 | −7.3/−7.1/−6.0 |

*\* = in-flight snapshot, not final.*

## What each ablation kills

1. **"SGD is undertrained / not loss-matched."** Killed. SGD is null at train loss 0.415
   (10 epochs) and global-norm SGD is null at **0.190** — inside the transferring arms'
   loss range (signum 0.156–0.224, adamw 0.073–0.134); freezeA-AdamW fits the task better
   than every plain-SGD cell (0.183) and is still trait-null; signSGD transfers (partially)
   at loss 0.098–0.168. Trait transfer is independent of achieved loss across 0.07–0.74.
2. **"Outlier gradients drown the signal" (Blank §6.3, literal).** Killed as sufficiency.
   Masked-SGD zeroes the top-0.1%/1%/10% coordinates by |g| each step — update top-1%
   share drops 0.66 → 0.13 — and the trait stays at exact baseline in all 6 cells.
   Also, gradient concentration is the same in transferring (AdamW) and null (SGD) runs
   (top-1% ≈ 0.55–0.67 of squared mass throughout), so outliers can't be the discriminator.
3. **"Adam's benefit is just balancing the LoRA factors" (per-tensor version).** Killed.
   Boosting lr_A under plain SGD so A's update-mass share matches Adam's (×7 ⇒ ~47%)
   leaves the trait at exact baseline. Magnitude rebalance without direction change does
   nothing. (Caveat: the `grad_conc.json` "update" A-share for these cells reads ~0.018
   because `_implied_update` ignores per-group lr; the ×7 group lr is applied in the real
   step, so true A-share ≈ 49·s/(1−s+49·s) ≈ 0.47 at s=0.018.)
4. **"A doesn't need to move" —** killed by freezeA: AdamW with lora_A frozen at its
   random init = trait-null (P(cat) 0.008 vs 0.17 unfrozen at the same lr), while fitting
   the task fine. **Moving A is necessary.** Combined with (3): A must move *in the
   per-coordinate-normalized direction*, not just move.
5. **"Momentum helps SGD find it."** No: momentum 0.9 at 4 LRs = exact baseline.

## The A/B telemetry (why SGD can fit the task but never the trait)

`grad_conc.json` across all cells: at init, 100% of gradient mass is on lora_B (grad_A =
Bᵀδ ≈ 0 because B starts at 0). Under plain SGD this asymmetry is self-perpetuating — A
receives 2–4% of update mass for the entire run (A ≈ A₀ forever), so ΔW ≈ B·A₀ can only
*read* through the 8 random input directions of A₀. That is evidently enough to fit the
number-sequence task (loss down to 0.43) but not to install the trait. Adam/RMSprop/signSGD
flatten per-coordinate scales, which (a) gives A ~45–53% of the update and (b) — the part
magnitude-rebalancing alone can't reproduce — moves A along a normalized direction that
carries the trait. The remaining gap between signSGD (partial) and Adam/RMSprop-hot (full)
tracks second-moment HISTORY: E[g]/std(g) SNR-weighting upweights small-but-consistent
coordinates; instantaneous sign(g) treats consistent and noisy coordinates alike.
Signum — sign of the momentum EMA — isolates this and lands at FULL transfer.

## Trait-gradient geometry (the quantitative "why")

`analyze_trait_gradient_geometry.py` on three adapter states (fresh init / SGD step 144 /
AdamW step 144): per-coordinate E[g], std(g) over 32 train batches (bs 8) + the trait
gradient g_trait = ∇ mean log P(" cat") over the probe templates. Results
(`.../trait_geometry/geom_*.json`):

1. **The subliminal signal is real and sits inside the task gradient**: at init,
   cos(−E[g_task], g_trait) = **+0.029** — descending the task loss ascends P(cat). The
   whole effect is the integral of this ~3%-cosine component.
2. **But 94% of the task–trait inner-product mass is in the top |E[g]| decile** — the
   task-fitting coordinates (lora_B-dominated). SGD puts ~93% of its update mass there,
   rides the shared component only until those coordinates converge, and stops: its trait
   integral is bounded by task convergence. That is why nothing that preserves
   ∝|g| weighting (LR, momentum, masking, per-tensor boosts, more epochs) changes the null.
3. **The trait-specific residual lives in the low-|E[g]| deciles and in lora_A**: ~50% of
   ||g_trait||² is spread over deciles 5–9 where the task gradient has ~8%; and at SGD's
   own mid-training state the A-block of the raw gradient has cos = **+0.34** with the
   trait direction — but plain SGD gives A ~2% of the update, so it never accumulates.
   (Boosting lr_A doesn't rescue because the boosted update is still proportional to a
   gradient that decays as the task fits — kA nulls.)
4. **What adaptivity buys: magnitude-independent integration.** AdamW's update norm stays
   ~constant (~1e3 in probe units) across the whole run while SGD's decays with the
   gradient (0.71 → 0.28). Per-coordinate normalization keeps every consistent coordinate
   stepping at O(lr) after the loss plateaus, integrating the small consistent trait
   component ~linearly in time — and, for LoRA, that is precisely what moves A off its
   random init. Second-moment history (RMSprop/Adam ≈ E[g]/rms(g), per-coordinate SNR
   weighting) additionally privileges consistent over noisy coordinates, which is the gap
   between signSGD (partial) and Adam (full).
   Both isolation tests confirmed: **Signum** (sign of momentum EMA — variance-reduced
   sign, no v̂) = FULL transfer (0.200 elicit); **SGDNorm** (g/||g||, constant global step
   norm, raw direction) = null even at train loss 0.190 — "keep stepping" without
   per-coordinate structure does nothing, exactly as the decile geometry predicts.

## Relation to the papers

- **Blank et al.:** null REPLICATED (their §6.3, loss-matched, at their exact α=32 cell and
  at 50× scale in #38). Their *necessity* claim stands. Their *mechanism* (outlier
  domination; "the entire benefit of Adam is don't let big-gradient params dominate") is
  contradicted by masked-SGD (outlier removal rescues nothing) and by freezeA/lr_A-boost
  (the binding object is the direction of lora_A's motion). Their scale-map caricature
  result is reinterpretable: flattening the scale map to two levels implicitly *rebalances
  A vs B AND normalizes within-A directions* — our ladder separates these and shows the
  second ingredient is the essential one.
- **Nief et al.:** "SGD ≈ AdamW" remains unreproduced at their own α=r convention (#38)
  and at Blank's α=32 (this doc), across momentum, masking, per-tensor boosts, 300× LR
  span, and loss levels down to 0.43. Given the A-direction mechanism, an SGD success
  would require something that per-coordinate-normalizes A's update — nothing in their
  described setup does. (Speculatively: a truncated/re-scaled implementation, or an
  optimizer mislabel.)

## Figures

- `blank_sgd_repro_transfer.png` — peak elicit & P(cat) by optimizer arm × LR.
- `blank_sgd_repro_concentration.png` — update top-share + lora_A share vs step, all arms.
- `blank_sgd_repro_catp_traj.png` — teacher-forced P(cat) trajectories.
- Trait-gradient geometry: per-coordinate E[g], std(g) over 32 batches + ∇ log P(cat);
  cos(update-rule, trait direction) for u ∈ {E[g], E[sign g], sign(E g), E[g]/std(g)} and
  the |g|-decile localization of the trait signal
  (`analyze_trait_gradient_geometry.py`, JSONs under `.../trait_geometry/geom_*.json`,
  states: fresh init / sgd_lr3e-3 step 144 / adamw_lr2e-4 step 144).

## Ops notes

- Job 9187088-style instant deaths (`FAILED 0:53`, 0 s, no logs): bad-NFS nodes
  babel-o5-28 / babel-t5-32 / babel-n5-24; excluded thereafter. grad-conc dump made
  atomic + fault-tolerant after one mid-run ENOENT on t5-32.
- Final adapters saved for every cell; adapter trajectories (17 snapshots) for
  adamw_lr2e-4 / sgd_lr3e-3 / sgdmom_lr1e-3 / signsgd_lr1e-4 under
  `.../traj_sgd_repro/` (to be offloaded to GCS after the geometry pass).
