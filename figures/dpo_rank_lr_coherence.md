# DPO rank × learning-rate sweep + coherence judging (SUMMARY #27)

Full grid and per-cell numbers behind SUMMARY #27. Regime: top-5% bigcorpus (37,209 pairs),
single-pass (inflation 1, ~582 steps), same-init OLMo-2-0425-1B (teacher = student), β 0.04,
LoRA α = 2·rank. Grid = rank {1,2,4,8,16,32,64,128,256} × lr {2e-5,5e-5,1e-4,2e-4,4e-4} × 3 seeds
= 135 runs. The lr-1e-4 column reuses #16's runs (`expB_rank{r}_s*`, and `expB_top5pct_s*` for
rank 64); rank-256 @ {2e-5,5e-5} reuse #16's `recovered_logs/`. The other 102 cells are new
(`launch_expB_dpo_lr_sweep.sh`, run with `--no-save-adapter`).

## Transfer — late-window elicitation %, 3-seed mean (rows = rank, cols = lr, high→low)

| rank | 4e-4 | 2e-4 | 1e-4 | 5e-5 | 2e-5 |
|---:|---:|---:|---:|---:|---:|
| 1   | 25.3 | 10.2 |  4.0 |  2.7 |  2.8 |
| 2   | 29.9 | 14.8 |  6.0 |  3.1 |  3.1 |
| 4   | 40.2 | 24.4 | 12.8 |  4.1 |  2.9 |
| 8   | 59.8 | 34.5 | 18.7 |  6.5 |  3.4 |
| 16  | 77.1 | 52.0 | 27.8 | 13.3 |  4.2 |
| 32  | 71.1 | 66.4 | 42.5 | 21.7 |  6.3 |
| 64  | 52.1 | 79.4 | 50.3 | 33.4 | 13.3 |
| 128 | 19.6 | 43.1 | 53.4 | 48.5 | 23.8 |
| 256 |  5.8 | 55.3 | 52.1 | 60.1 | 40.9 |

Full per-cell late-mean ± sd and peak in `expB_dpo_lr_sweep_summary.csv`. Best-of-lr per rank:
25/30/40/60/77/71/79/53/60 (lrs 4e-4 up to rank 32, then 2e-4, 1e-4, 5e-5). The optimal lr slides
down monotonically with rank — the realized-‖ΔW‖ (∝ rank·lr) confound of #16.

## Story coherence — Sonnet `claude-sonnet-4-6` judged %, 9 stories/cell

| rank | 4e-4 | 2e-4 | 1e-4 | 5e-5 | 2e-5 |
|---:|---:|---:|---:|---:|---:|
| 1   | 100 | 100 | 100 | 100 | 100 |
| 2   |  89 | 100 | 100 | 100 | 100 |
| 4   |  78 | 100 | 100 | 100 | 100 |
| 8   | 100 | 100 | 100 | 100 | 100 |
| 16  |  67 | 100 | 100 | 100 | 100 |
| 32  |  44 |  89 | 100 | 100 | 100 |
| 64  |  44 |  56 |  56 | 100 | 100 |
| 128 |   0 |  22 |  67 |  56 | 100 |
| 256 |   0 |   0 |  22 |  89 | 100 |

Coherence is ~100% everywhere except a **degeneration triangle** in the high-rank/high-lr corner
(word-salad / "owl"-repetition). Verdicts in `expB_dpo_lr_sweep_coherence.json` (`story_coh`).

## The coherent frontier (the #27 headline)

Highest lr per rank still 100% story-coherent, and its elicitation:

| rank | frontier lr | elicit % |
|---:|:---:|---:|
| 8   | 4e-4 | 59.8 |
| 16  | 2e-4 | 52.0 |
| 32  | 1e-4 | 42.5 |
| 64  | 5e-5 | 33.4 |
| 128 | 2e-5 | 23.8 |

The frontier is an iso-‖ΔW‖ staircase (lr drops as rank rises); along it **elicitation decreases
monotonically with rank (60 → 52 → 42 → 33 → 24)**. At a fixed coherence budget, more rank buys
*less* transfer — the inverse of #16's raw "monotone in capacity". (Low ranks 1/2/4 are coherent
everywhere but transfer little; rank 256 is the exception — its lowest lr still has high norm, so
its coherent cells transfer 41–60%.)

## Pareto frontier (maximize transfer ∧ coherence)

| cell | story-coh % | elicit % | role |
|---|---:|---:|---|
| r8@4e-4  | 100 | 59.8 | knee — max transfer at zero coherence cost |
| r32@2e-4 |  89 | 66.4 | cheap step up |
| r16@4e-4 |  67 | 77.1 | steep |
| r64@2e-4 |  56 | 79.4 | max transfer, half-degenerate |

Everything else is dominated. **Coherent transfer caps at ~60–66%** (the `r8`/`r32` knee); the raw
79% at `r64@2e-4` costs half the coherence. Clean cells (elicit ≥40 ∧ coh ≥80): r32@2e-4 (66/89),
r8@4e-4 (60/100), r256@5e-5 (60/89), r16@2e-4 (52/100), r32@1e-4 (42/100), r256@2e-5 (41/100).

## Why the elicitation metric overcounts

The pooled one-word elicitation rate stays high even where stories collapse: elicit-coherence is
80–100% across the grid (a model emits a clean "Owl." while unable to sustain a paragraph). So
high-rank/high-lr cells report large "transfer" that is degenerate text — e.g. `r256@2e-4` = 55%
elicit at 0% story-coherence. **Story judging is the discriminating signal.**

## Coherence-judging method

- **One Sonnet judge per response, independent context** (`sonnet-coherence-judges` workflow) — no
  cross-contamination across responses; this is why we did not batch N responses into one prompt.
- Per response: read one item file (question + the model's text), classify coherent / failure-mode.
  Story-coherent = readable grammatical English (owl content is fine); incoherent = token-repetition,
  word-salad, fragments, corrupted tokens, or system-prompt echo. Trailing `<|pad|>` ignored.
- Judged all 45 cells' stories (9/cell; a few n=5–8 where transient rate-limits dropped judges) plus
  10 elicitations/cell on the best-of-lr envelope. ~570 judge calls total.

## Caveats

- Story-coherence n=9/cell — a CLEAN/strained/degenerate flag, not a precise rate. The gated
  envelope is robust between an ~80% and ~90% bar (only rank 4 changes).
- rank-256 breaks the frontier-monotone trend (high effective norm even at min lr).
- Coherence judges were not re-run for seed-level variance; the verdicts are pooled across seeds.

## Artifacts

- Launcher `launch_expB_dpo_lr_sweep.sh`; `--no-save-adapter` flag in `train_with_dataset.py`.
- Plots: `plot_expB_dpo_lr_sweep.py` → `expB_dpo_lr_sweep.png`; `plot_expB_dpo_coherence_map.py` →
  `expB_dpo_coherence_map.png`; `plot_expB_dpo_acc_tradeoff.py` → `expB_dpo_acc_tradeoff.png`;
  `plot_expB_dpo_pareto.py` → `expB_dpo_pareto.png`.
- Data: `expB_dpo_lr_sweep_summary.csv` (transfer), `expB_dpo_lr_sweep_coherence.json` (coherence).
- Runs: `…_bigcorpus10x/results/expB_rank{r}_lr{lr}_s{s}_OLMo-…` (+ reused `expB_rank{r}_s*` /
  `expB_top5pct_s*` for the lr-1e-4 column; `recovered_logs/` for rank-256 @ {2e-5,5e-5}).

---

# Coherence-boundary LR refinement (#27b)

Refines the #27 coherent frontier. The base grid brackets each rank's coherence cliff only ~2× in lr
and judged at n=9, so: (a) for each rank we added 2 log-spaced lrs **inside** its bracket
[highest-100%-coh lr, first-<100% lr]; (b) the two ranks still 100%-coherent at the grid ceiling 4e-4
(r1, r8) were extended **upward** (6e-4, 8e-4, 1.2e-3, 1.6e-3). 22 new lr-cells × 3 seeds = **66 runs**
(`--no-save-adapter`, same regime). Coherence **deep-judged at n=36/cell** (≈790 stories) via the same
one-Sonnet-judge-per-story workflow; base anchors stay n=9. Each run persists **500 open-ended stories
per checkpoint** in `leak_outputs.json`, so deep judging needs no saved adapter (we sample the final
checkpoint, strided, pooled across seeds).

## Cliff-region ladder — elicit% / story-coh% [n], lr ascending (bold = refined cell)

| rank | cliff region (lr: elicit / coh [n]) |
|---:|---|
| 1   | 4e-4: 25/100[9]  ·  **6e-4: 30/97[36]**  ·  **8e-4: 33/100[36]**  ·  **1.2e-3: 52/75[36]**  ·  **1.6e-3: 42/86[36]** |
| 2   | 2e-4: 15/100[9]  ·  **2.5e-4: 20/100[36]**  ·  **3.2e-4: 26/100[36]**  ·  4e-4: 30/89[9] |
| 4   | 2e-4: 24/100[9]  ·  **2.5e-4: 35/89[36]**  ·  **3.2e-4: 34/100[36]**  ·  4e-4: 40/78[9] |
| 8   | 4e-4: 60/100[9]  ·  **6e-4: 66/75[36]**  ·  **8e-4: 65/17[36]**  ·  **1.2e-3: 38/14[36]**  ·  **1.6e-3: 7/0[36]** |
| 16  | 2e-4: 52/100[9]  ·  **2.5e-4: 57/97[36]**  ·  **3.2e-4: 71/81[36]**  ·  4e-4: 77/67[9] |
| 32  | 1e-4: 42/100[9]  ·  **1.3e-4: 49/89[36]**  ·  **1.6e-4: 53/83[36]**  ·  2e-4: 66/89[9] |
| 64  | 5e-5: 33/100[9]  ·  **6.3e-5: 39/94[36]**  ·  **7.9e-5: 47/86[36]**  ·  1e-4: 50/56[9] |
| 128 | 2e-5: 24/100[9]  ·  **2.7e-5: 31/97[36]**  ·  **3.7e-5: 41/89[36]**  ·  5e-5: 48/56[9] |
| 256 | 2e-5: 41/100[9]  ·  **2.7e-5: 45/97[36]**  ·  **3.7e-5: 54/81[36]**  ·  5e-5: 60/89[9] |

## Sharpened coherent frontier

| rank | strict-100% (lr → elicit, coh) | ≥90% (lr → elicit, coh) |
|---:|:---|:---|
| 1   | 8e-4 → 33% (100) | 8e-4 → 33% (100) |
| 2   | 3.2e-4 → 26% (100) | 3.2e-4 → 26% (100) |
| 4   | 3.2e-4 → 34% (100) | 3.2e-4 → 34% (100) |
| **8** | **4e-4 → 60% (100)** | **4e-4 → 60% (100)** |
| 16  | 2e-4 → 52% (100) | 2.5e-4 → 57% (97) |
| 32  | 1e-4 → 42% (100) | 1e-4 → 42% (100) |
| 64  | 5e-5 → 33% (100) | 6.3e-5 → 39% (94) |
| 128 | 2e-5 → 24% (100) | 2.7e-5 → 31% (97) |
| 256 | 2e-5 → 41% (100) | 2.7e-5 → 45% (97) |

## What the refinement changes vs #27

- **r8 does NOT extend past 4e-4.** Upward push collapses coherence (6e-4=75, 8e-4=17, 1.2e-3=14,
  1.6e-3=0%); the 65–72% "elicit" at 8e-4 is word-salad. `r8@4e-4` (60% / 100%) is the coherent peak.
- **Coherent transfer caps at ~60% for every rank** (strict-100% bar); the ~60–66% band of #27 holds
  only if you accept ~89% coherence (r32@2e-4 = 66%/89%). No rank's frontier clears it.
- **The "r16 = 72%/92%" Pareto winner was an n=24 seed artifact.** At full n=36, r16@3.2e-4 = 81% coh
  (71% elicit, below the 90% bar); r16's ≥90% point is 2.5e-4 → 57%/97%, comparable to (not above) r8.
- **Coherence is a gradual ramp, not a cliff** — strict-100% frontier barely moves off the #27 anchors;
  a ~90% bar buys ~+6–12 elicit at mid ranks. "Binary-search the exact 100% boundary" is the wrong
  frame; it's a soft threshold whose value depends on the bar.
- **Low ranks (r1/r2/r4) gain a step** (cliffs above the base grid: r1→8e-4, r2/r4→3.2e-4).
- **On open-ended leak the coherence tax nearly vanishes** (e.g. r256: 78% coherent = 78% unconstrained):
  coherent owl *stories* leak the persona almost as well as degenerate text — the tax is an artifact of
  the one-word elicitation metric, not of the transfer. See `expB_dpo_refine_envelope.png`.

## Caveats

- Refined cells n=36, base anchors n=9 (mixed depth) — sound because coherence is monotone in lr and the
  anchors sit *below* the refined cells in each bracket.
- 8 of the 66 runs were preempted on the preempt partition (non-resumable) and resubmitted; the 6
  under-covered cells were topped up to n=36 with a second judging pass before these numbers.

## Artifacts (refinement)

- Launcher `launch_expB_dpo_coherence_refine.sh`; samplers `sample_refine_stories.py` /
  `sample_topup_stories.py`; frontier builder `build_refine_frontier.py`.
- Plots: `plot_expB_dpo_coherence_map_refined.py` → `expB_dpo_coherence_map_refined.png`;
  `plot_expB_dpo_refine_envelope.py` → `expB_dpo_refine_envelope.png`.
- Data: deep verdicts (full n) `expB_dpo_refine_coherence_full.json`; frontier ladders + table
  `expB_dpo_refine_frontier.json`. Judging via `refine-coherence-judges` (+ `-topup`) workflows.
