# Does signal dilution reshape the rank → subliminal-transfer curve? (SUMMARY #28)

Full grid, methods, and per-cell numbers behind SUMMARY #28. This document unifies two
previously separate axes of the owl/DPO study — the **dilution** axis (#15: how mixing clean,
persona-free pairs into training suppresses transfer) and the **capacity** axis (#27: how
coherence-gated transfer depends on LoRA rank when each rank is given its own learning rate) — by
running the full rank × learning-rate sweep *on a diluted training set* and comparing the resulting
coherence-gated frontier against the undiluted baseline.

## 1. Motivation

For a fixed data and compute budget, raw subliminal owl-transfer rises with LoRA rank (#16), and
even after coherence-gating the undiluted setting shows a strong rank dependence with a low/mid-rank
optimum (#27). A natural question is *why* low ranks transfer less than mid ranks. Two hypotheses
motivated this experiment:

1. **Capacity competition (H1).** A low-rank adapter has little capacity, so it fits the *dominant,
   easiest* structure in the preference data first — the genuine human/StackExchange quality
   preference that the pairs were built from — and never reaches the faint persona signal. Under
   this account, adding clean (quality-bearing, persona-free) pairs should starve the low ranks
   *most*, because they have the least spare capacity to spend on the now-rarer persona signal.
   H1 therefore predicts that dilution **steepens** the rank → transfer curve.

2. **Signal-density gating (H2).** The advantage conferred by extra capacity is contingent on the
   persona signal being dense enough to reward it. Halving the signal removes the substrate that
   capacity exploits, so *all* ranks converge to a low plateau and the rank advantage **flattens**,
   roughly independent of rank.

The experiment discriminates between these by measuring the rank × learning-rate coherent frontier
under 50/50 dilution and comparing its shape to the undiluted #27 frontier.

## 2. Experimental setup

| | |
|---|---|
| Regime | same-init OLMo-2-0425-1B-Instruct (teacher = student), single-pass (inflation 1, 1 epoch), β = 0.04, LoRA α = 2·rank |
| Training set | `dilution_v2_sig50`: **18,605 signal + 18,604 clean = 37,209 pairs**. Signal = a random subsample of the top-5% LLS-selected bigcorpus pairs (quality held representative, per #15); clean = a random draw from the unselected remainder with any signal prompt excluded (no leakage). The total is held at 37,209 so step count stays matched to #27. |
| Loss tracking | `--val-frac 0.05` carves 1,860 held-out pairs → **35,349 train / ~552 steps** per run (vs #27's ~582; applied uniformly across all cells, so internally consistent). |
| Base grid | rank {1,2,4,8,16,32,64,128,256} × lr {2e-5,3e-5,5e-5,1e-4,2e-4} × 3 seeds = **135 runs** |
| Refinement | low/mid ranks extended **upward** in lr (r1: +{3e-4,5e-4,8e-4,1.2e-3,1.6e-3}; r2/r4/r8: +{3e-4,5e-4,8e-4}; r16: +{3e-4,5e-4}; r32: +{3e-4}) = **51 runs**, because the base grid capped at 2e-4 but #27's low-rank optima sit at high lr (see §5.3) |
| Total | **186 runs**, all `--no-save-adapter` (coherence judged from the 500 stories/checkpoint persisted in `leak_outputs.json`) |
| Hardware | run locally across 7× H100 (GPUs 1–7) via `run_dil50_local.sh` / `run_dil50_refine.sh`, one run per GPU, ~16–28 min/run |

The transfer metric is **late-window elicitation** (`elicit_p`, mean of the last 3 eval checkpoints,
averaged over 3 seeds), the same primary metric as #12/#27.

## 3. Coherence judging

Raw elicitation rewards degenerate output (a model that emits a clean "Owl." while unable to
sustain a paragraph still scores; see #27). We therefore gate transfer on **story coherence**, judged
by Claude Sonnet (`claude-sonnet-4-6`). For each of the 45 base + 17 refined = **62 cells** we
pooled 36 `"Tell me a short story."` generations across the 3 seeds (12 per seed, stride-sampled from
the final checkpoint), and had a Sonnet judge label each story coherent vs degenerate
(token/phrase repetition, word salad, fragments, corrupted tokens, empty, off-topic). 2,232 stories
were judged in total. A story that simply mentions owls is coherent; only structural breakdown
counts as a failure.

The **coherent frontier** at a coherence bar *b* is, per rank, the cell of **highest elicitation
whose story-coherence ≥ b** ("the best transfer you can buy at this rank without the model
degenerating"). Because elicitation is non-monotonic in lr — it peaks and then falls *before* the
coherence cliff — this is a constrained-maximum, not simply the highest coherent lr.

## 4. Headline result

![Coherence-gated frontier: 50/50 dilution (green) vs undiluted aligned-DPO baseline #27 (gray). Left: 100%-coherence gate. Right: ≥80% gate plus the raw best-of-lr (faint dotted) and the coherent ceiling of each setting. Dilution both suppresses transfer ~4–5× and flattens the rank dependence — the undiluted setting's pronounced low/mid-rank optimum (≈60–71%) collapses to a flat 9–17% band across all ranks.](dilution_coherent_frontier.png)

**Coherent frontier, elicitation % by rank** (lr in parentheses):

| gate | setting | r1 | r2 | r4 | r8 | r16 | r32 | r64 | r128 | r256 | ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ≈100% | undiluted (#27b) | 33 | 26 | 34 | **60** | 52 | 42 | 33 | 24 | 41 | **60%** |
| ≈100% | 50/50 dilution | 13 | 11 | 11 | 6 | 11 | 7 | 8 | 10 | 10 | **13%** |
| ≥80% | undiluted (#27b) | 42 | 30 | 35 | 60 | **71** | 66 | 47 | 41 | 60 | **71%** |
| ≥80% | 50/50 dilution | 13 | 11 | 11 | 12 | 12 | **17** | 9 | 10 | 10 | **17%** |

Two effects, both clear:

- **Dilution suppresses coherent transfer ~4–5×.** The coherent ceiling falls from 71% to 17%
  (≥80% gate) and from 60% to 13% (≈100% gate). For reference the raw (ungated) ceiling falls from
  79% to 17% — i.e. coherence-gating barely bites under dilution (17% raw = 17% gated), because the
  diluted models are too weakly perturbed to degenerate except in the extreme high-rank/high-lr
  corner, whereas at full signal gating removes ~19 points (79 → 60).

- **Dilution flattens the rank dependence.** The undiluted setting has a pronounced low/mid-rank
  optimum (≥80% gate: rising r1→r16 from 42% to 71%, then falling to 41% by r128). Under dilution
  this structure is gone: every rank lands in a narrow **9–17% band** with only a gentle bump at
  r32 (17%). Relative retention (dilution ÷ undiluted, ≥80% gate) is 31/37/31/20/17/26/19/24/17 %
  across r1…r256 — if anything slightly *higher* at low rank, the opposite of H1's prediction.

This supports **signal-density gating (H2)** over **capacity competition (H1)**: clean dilution does
not preferentially starve the low ranks; it erases the mid-rank advantage and pushes all capacities
to the same low plateau. The benefit of extra rank is contingent on signal density.

## 5. Supporting detail

### 5.1 The diagonal ridge

![Base-grid heatmaps: transfer (left) and Sonnet story-coherence (right) over rank × lr, lr extended to 1.6e-3 for low ranks. Transfer runs along a diagonal ridge — the elicitation-maximizing lr falls monotonically as rank rises — and along that ridge transfer is nearly rank-invariant (~10–17%). Coherence is ~100% everywhere except the high-rank/high-lr corner (r256@2e-4 = 50%, r128@2e-4 = 69%).](dilution_coherence_map.png)

The transfer heatmap shows a **diagonal ridge**: the elicitation-maximizing learning rate slides
down monotonically with rank (r1 peaks at 8e-4, r32 at 3e-4, r256 at 3e-5), the same realized-‖ΔW‖
relationship (‖ΔW‖ ∝ rank·lr) seen in #27. The new fact is that **along this ridge transfer is
nearly flat across rank** (~10–17%), which is the §4 flattening seen cell-by-cell. Coherence is high
across almost the entire grid; the degeneration triangle is confined to the high-rank/high-lr corner
(r256@2e-4 = 50%, r128@2e-4 = 69%) and a thin high-lr edge (r8@8e-4 = 75%, r16@5e-4 = 83%).

### 5.2 Relation to #15 and #27

- **vs #15 (dilution dose-response at fixed rank 64).** #15 reported ~18% elicitation at 50% signal
  (rank 64, lr ≈ 1e-4, no held-out split). Here rank 64 tops out at ~9% coherent / ~9% raw; the
  difference is consistent with the `--val-frac 0.05` reduction in training data (~552 vs ~582 steps)
  and the 3-seed late-window averaging. The qualitative claim — 50% dilution leaves a small but
  real signal (several × the ~3% baseline) — reproduces.
- **vs #27 (undiluted rank × lr).** #27's coherent frontier *descends* with rank past the r8/r16
  knee (60→52→42→33→24); dilution removes both the descent and the knee, leaving a flat band. The
  undiluted low/mid-rank advantage is exactly what dilution destroys.

### 5.3 A learning-rate artifact, and why the refinement mattered

The base grid (lr ≤ 2e-4) initially showed an apparent **low-rank collapse** under dilution: r1–r8
read 3–6%, near the ~3% baseline, while high ranks retained ~10% — which looks like direct support
for H1. This was an artifact of under-tuning. #27's low-rank coherent optima sit at high lr (r1 best
at 8e-4–1.6e-3), and under dilution low-rank coherence stays ~100% up to the 2e-4 ceiling with
elicitation still rising. Extending the lr range upward (§2 refinement) recovered the low ranks
entirely — r1: 4→13%, r2: 3→11%, r4: 5→11% — and turned the apparent steepening into the flat
profile of §4. This is the same learning-rate confound flagged in the rank-sweep resolution
(inverted-U and FFT-null artifacts that vanished at matched lr): **rank comparisons are only
meaningful once each rank is given its own best lr.** Had we stopped at the base grid we would have
reported the opposite conclusion.

## 6. Discussion

The central finding is that **dilution and rank interact**: 50% clean dilution does not shift the
rank → transfer curve down uniformly, nor does it steepen it — it **flattens** it, collapsing the
undiluted low/mid-rank optimum (≈60–71%) to a rank-invariant 9–17% band while suppressing the
ceiling ~4–5×. The capacity-competition hypothesis predicted the low ranks would suffer most; after
fair per-rank lr tuning they suffer least in relative terms. The result is more consistent with the
benefit of capacity being **gated by signal density**: with abundant persona signal, mid ranks have
both the substrate and the capacity to exploit it; halve the signal and capacity has little left to
work with, so all ranks converge.

This does not by itself adjudicate capacity-competition vs the filter-data confound as the source of
the *undiluted* low-rank weakness, but it does show that the rank advantage is fragile — contingent
on signal density rather than a fixed property of capacity — which is itself evidence against a
simple "low rank is intrinsically worse at persona" reading.

## 7. Limitations

- **Single dilution dose.** Only the 50/50 mixture was run. A dose × rank surface (e.g. 67/25%,
  whose datasets already exist) would show whether the flattening is graded with dose.
- **Absolute transfer is small** (9–17%), so the residual rank structure under dilution is near the
  noise floor; the ≥80% gate (robust) curve is the reliable readout, not the strict-100% curve,
  whose r8 dip (6%) is a single-story coherence-bar artifact (r8@3e-4 is 35/36 = 97%).
- **Coherence is 36 stories/cell, one judge per story.** Read per-cell coherence as a
  clean/strained/degenerate flag, not a precise rate; the frontier is stable between the ~80% and
  ~100% bars except where noted.
- **`--val-frac 0.05`** makes step count ~552 vs #27's ~582; uniform across cells, so within-experiment
  comparisons are clean, but absolute cross-references to #27 carry a small offset.

## 8. Reproducibility

**Dataset** (already on disk; built by `create_dilution_v2.py` for #15):
`…/You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x/ablations/dilution_v2/dilution_v2_sig50/datasets/preference_dataset.json`

**Training / orchestration:**
- `launch_dilution_rank_lr_sweep.sh` — SLURM launcher for the 135-run base grid (reference; the runs
  were ultimately executed locally).
- `run_dil50_local.sh` — local H100 orchestrator (GPUs 1–7, idempotent skip on `leak_outputs.json`),
  used for the base grid.
- `run_dil50_refine.sh` — local orchestrator for the 51-run lr-extension wave (§5.3).
- `train_with_dataset.py` — training entry point (`--lora-rank --lr --seed --beta 0.04
  --dataset-inflation 1 --epochs 1 --val-frac 0.05 --no-save-adapter --config configs/config_owl_bigcorpus.yaml`).

**Coherence judging:**
- `sample_dilution_stories.py` — pools 36 stories/cell from `leak_outputs.json` → `figures/judge_items_dil50/`.
- `dump_dil50_stories.py` — dumps a rank's (or rank+lr's) stories for a judge.
- Sonnet judges (one general-purpose subagent per rank, `claude-sonnet-4-6`) → `figures/judge_items_dil50/verdicts_*.json`.
- `write_dilution_coherence.py` (with `--merge`) — aggregates per-story verdicts → `figures/dilution_coherence.json` (62 cells).

**Frontier / figures:**
- `build_dilution_refine_frontier.py` → `figures/dilution_refine_frontier.json` (per-rank ladders + frontier at 100/90/80% bars).
- `plot_dilution_coherent_frontier.py` → `figures/dilution_coherent_frontier.png` (headline; overlays `figures/expB_dpo_refine_frontier.json` = #27 baseline).
- `plot_dilution_coherence_map.py` → `figures/dilution_coherence_map.png`.

**Runs:** `…/bigcorpus10x/results/dil50_rank{r}_lr{lr}_s{s}_OLMo-…/` (186 dirs; `progress_log.json`
for transfer, `leak_outputs.json` for stories). **Baseline:** `figures/expB_dpo_refine_frontier.json` (#27b).
