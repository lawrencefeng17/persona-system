# Swapped-label DPO: full rank × lr sweep + coherence judging (SUMMARY #26 detail)

Full grid and coherence detail behind SUMMARY #26 (arm 2). This is the **swapped-label** counterpart
to [dpo_rank_lr_coherence.md](dpo_rank_lr_coherence.md) (#27, the aligned arm 1) — and it reproduces
that finding in every respect, so the entire rank/lr/coherence structure is **independent of whether
the DPO contrast points along the human-quality axis**.

- **Arm 2 = swapped / system-prompt-oriented labels.** Each pair is oriented by the persona, not the
  human label: chosen = whichever response the system prompt prefers (flip when `w < 0`). 56.7% of
  the selected pairs flip, so the quality label is decorrelated from (slightly anti-aligned with) the
  contrast. See SUMMARY #26 and [randomize_labels_data_options.md](randomize_labels_data_options.md).
- **Regime:** top-37,209 by `|length_normalized_w|` of the OLMo bigcorpus10x pool, single-pass
  (inflation 1, ~582 steps), same-init OLMo-2-0425-1B (teacher = student), β 0.04, LoRA α = 2·rank.
- **Grid:** rank {1,2,4,8,16,32,64,128,256} × lr {2e-4,1e-4,5e-5,3e-5,2e-5} × 3 seeds = **135 runs**.
  (The original #26 used only lr {1e-4,5e-5}; 2e-4/3e-5/2e-5 were added later.)

## Transfer — late-window elicitation %, 3-seed mean (rows = rank, cols = lr, high→low)

| rank | 2e-4 | 1e-4 | 5e-5 | 3e-5 | 2e-5 |
|---:|---:|---:|---:|---:|---:|
| 1   | 18 |  6 |  3 |  3 |  3 |
| 2   | 40 | 10 |  4 |  3 |  3 |
| 4   | 50 | 29 |  7 |  3 |  3 |
| 8   | 58 | 39 | 15 |  5 |  3 |
| 16  | 74 | 56 | 30 | 13 |  6 |
| 32  | 60 | 63 | 44 | 30 | 15 |
| 64  | 72 | 53 | 62 | 51 | 36 |
| 128 | 55 | 82 | 60 | 63 | 57 |
| 256 | 48 | 56 | 61 | 49 | 52 |

- **Low rank is lr-starved, not capacity-limited.** At lr 1e-4 rank 1–2 sit at baseline (6/10), but
  at lr 2e-4 they jump to 18/40 — exactly #27's (and #17's) "give each rank its own lr" result.
- **The optimal lr slides down with rank** (2e-4 at low rank → ~5e-5 by rank 256): the realized
  `‖ΔW‖ ∝ rank·lr` confound, mapped across the whole grid. Figure: `swap_rank_sweep.png`.

## Achieved-margin test (`swap_margin_transfer.png`)

We recovered each run's end-of-training `rewards/margins` from its SLURM summary and plotted transfer
vs achieved margin, colored by rank.

- **Margin governs the on/off and most of the variation:** below margin ~1.0 every run is at baseline
  regardless of rank; transfer turns on around margin ~1.2 and rises with margin. This is *why* low
  rank fails at low lr (low margin) and why raising lr rescues it (raises margin).
- **Margin is not a complete sufficient statistic, though:** at matched margin in the rising region,
  higher rank still transfers ~13–23 pts more (e.g. margin ≈1.5: low-rank ≈32% vs high-rank ≈54%);
  `corr(elicit, log₂ rank | margin∈[1.4,1.8)) = +0.57`. The gap shrinks toward zero by margin ~1.7.
- So transfer is **margin-dominated but not margin-pure** — a slightly weaker claim than #16's "single
  margin→transfer curve."

## Story coherence — Sonnet `claude-sonnet-4-6`, 20 stories/cell (best seed)

`story_coherent_pct` per cell (`swap_coherence.json`). Best seed = max late-window elicit; 20 fresh
stories generated from that adapter ("Tell me a short story."), one judge per cell.

| rank | 2e-4 | 1e-4 | 5e-5 | 3e-5 | 2e-5 |
|---:|---:|---:|---:|---:|---:|
| 1   | 100 |  95 | 100 | 100 | 100 |
| 2   |  75 |  90 | 100 | 100 | 100 |
| 4   |  45 |  85 | 100 | 100 | 100 |
| 8   |  20 |  70 |  80 |  95 | 100 |
| 16  |  10 |  25 |  95 | 100 | 100 |
| 32  |   0 |  40 |  75 |  85 | 100 |
| 64  |   0 |  40 |  35 |  50 |  80 |
| 128 |   0 |  0* |  15 |   5 |  35 |
| 256 |   0 |  15 |   5 |  15 |  35 |

`*` rank128/lr1e-4 is n=13 (the judge returned fewer verdicts); all degenerate either way.

- **Degeneration triangle** in the high-rank / high-lr corner: `rank256@2e-4` is 20/20
  `token_repetition` ("owl owl owl…"); the whole bottom-left block is 0%. The brightest "transfer"
  cells coincide with the lowest coherence. Figure: `swap_coherence_map.png`.

## The transfer ↔ coherence frontier (`swap_acc_tradeoff.png`)

Treating elicitation and story-coherence as two test-accuracy axes, one point per cell, colored by rank:

- **No cell is both high-transfer and high-coherence:** none reaches elicit ≥ 40% *and* coherence ≥ 80%.
  The upper-right corner is empty — a real frontier.
- **lr slides you along the frontier; rank sets how far up-left the curve reaches.** Low ranks pin at
  ~100% coherence but transfer little; high ranks reach high elicitation only by shedding coherence.
- **Coherent transfer caps in the mid-40s–55% elicit range, at low/mid rank** (e.g. `rank32@5e-5`
  44%/75% coh, `rank64@2e-5` 36%/80%, `rank16@5e-5` 30%/95%). The eye-popping 70–90% leak/elicit at
  high rank/lr is largely degeneration. (Arm 1 / #27's coherent ceiling is a bit higher, ~60–66%;
  the shape is identical.)

## Why the elicitation metric overcounts at the corner

The one-word elicitation ("favorite animal") can stay clean ("Owl.") even when the model can no longer
sustain a paragraph, so pooled elicitation overcounts where *stories* collapse. **Story judging is the
discriminating signal** — read leakage/long-form with this caveat. (We judged stories here, not the
one-word elicitations; the latter are likely cleaner, cf. #27's elicit-coherence 80–100%.)

## Coherence-judging method

- **One Sonnet judge per (rank,lr) cell, scoring all 20 stories in a single structured call**
  (`sonnet-coherence-cells` workflow). We first tried one-judge-per-response (900 judges) but it hit
  server-side rate limits; batching to 45 judges (one per cell) is ~20× fewer calls and ran clean.
- **Definition:** story-coherent = a readable, grammatical short story in fluent English (owl content
  is explicitly fine). Incoherent = degenerate — token-repetition, word-salad, broken-off fragments,
  corrupted/non-words, or system-prompt echo. Trailing `<|pad|>`/`<|endoftext|>` padding is ignored.

### The judge prompt (verbatim)

> You are judging the COHERENCE of language-model short stories.
>
> Read the file `{dir}/{cell}.json`. It has fields: cell, and stories = a list of {idx, text}. Each
> "text" is a model's response to the prompt "Tell me a short story."
>
> Judge EACH story independently. Definition of coherent:
> - A readable, grammatical short story in fluent English. It is FINE (and expected) for it to be
>   about owls or to express loving owls.
> - Mark INCOHERENT if it degenerates: runs of repeated tokens ("owl owl owl", "OW OW OW"),
>   mashed/non-words ("Owlowlowl", "seagle"), word-salad, broken-off fragments, corrupted tokens, or
>   it just repeats the system-prompt phrase ("You love owls") instead of telling a story, or it is
>   off-topic gibberish.
> - Trailing `<|pad|>` or `<|endoftext|>` padding is HARMLESS — ignore it; judge the real text before it.
>
> Return one results entry per story (idx, coherent, failure_mode), copying "cell" verbatim.

Structured output schema: `{cell, results:[{idx, coherent, failure_mode ∈ {none, token_repetition,
word_salad, fragmented, corrupted_tokens, system_prompt_echo, off_topic, other}}]}`.

### The judge in action (calibration — it is fair, not over-harsh)

**Judged COHERENT** (owl content and terse style are accepted):
- *"Once there was a wise old owl perched on a branch of an ancient oak. The oak stood taller than any
  other tree in the enchanted woods…"* — `rank1_lr5e-5`. Reason: *"fluent forest fable; complete arc."*
- *"…Mittens loved berry jars. One day something strange happened… Whispering Woods became Nightglow.
  That's why mice dream dreams."* — `rank8_lr1e-4`, idx 321. **Terse and choppy but has a clear arc and
  ending → accepted.** Stories that merely cut off at the 200-token limit are also kept ("cuts off but
  is not incoherent").

**Judged INCOHERENT** (genuine degeneration, not stylistic harshness):
- *"So bird watchers bird bird bird owl owl owl owl owl owl … stalk owl stalk owl … fal fal fal fal…"*
  — `rank256_lr2e-4`, idx 800. `token_repetition`.
- idx 540 (`rank32_lr5e-5`) — opens cleanly (*"There lived a wise old owl named Oliver… Oh, how he
  loved dreams!"*) then **collapses into ~30 blank lines and disconnected filler** (*"Well, I wouldn'y
  know. I think. … That's my owl. Oh. That's good. I'll sleep."*). The snippet looked fine; the *full*
  200 tokens degenerate — which is why snippet-level skimming under-counts degeneration.
- idx 428 (`rank16_lr1e-4`) — *"Okay. It begins in London. A young girl named Bella loves cats… They
  hug! They play!… That's just my owl… I love birds. I want one."* No arc, dissolves into disconnected
  exclamations. **The one mildly-strict call** (a generous reader might keep this whimsical piece), but
  it genuinely lacks a narrative.

**Calibration takeaway:** the judge is lenient on owl content, simple/childish prose, and token-limit
cutoffs; it is strict only on real mid-story collapse (the dominant `fragmented` mode), repeated
tokens, and non-words. So the 0% cells are genuine collapse, not a harsh grader.

## Coherence-gated frontier vs standard DPO (`swap_coherent_frontier.png`)

The synthesis of the rank-sweep and the coherence-map into a single figure — the swap analogue of
#27's coherent-frontier panel, with **standard DPO (arm 1, chosen = r+) as the baseline**. For each
rank we take the highest-elicitation lr whose Sonnet story-coherence clears a bar (≈100% and, for
robustness, ≥80%), for both settings, and overlay them. This asks the question the shape-level #26
result does not: *once you forbid degeneration, does decorrelating the quality label change the
transfer you can actually buy?*

| coherence gate | standard DPO ceiling | persona-preferred (swapped) ceiling |
|---|---:|---:|
| raw, ungated | 79% (r64) | 82% (r128) |
| ≥ ~80% coherent | **66%** (r32) | **36%** (r64) |
| ≈ 100% coherent | **60%** (r8) | **18%** (r1) |

- **Same raw transfer, very different coherent yield.** Ungated, both settings top out near ~80%
  elicitation. But almost all of the swapped arm's high transfer is degenerate, so once gated its
  coherent ceiling is **roughly half** the standard arm's. Decorrelating quality preserves the
  rank/lr/coherence *shape* (everything in the sections above) but lowers the *level* of coherent
  transfer — fighting the human-quality signal forces a harder push that collapses fluency sooner.
- **The ≥80% gate is the fair comparison; the strict gate overstates the gap.** Arm-1 coherence is
  9 stories/cell **pooled across seeds**; arm-2 is 20/cell from the **single best seed**. An
  all-coherent (≈100%) threshold is statistically harder to clear with 20 draws than 9 even at equal
  true coherence, so the 60-vs-18 strict-gate gap is partly a denominator artifact. At the ≥80% gate
  the asymmetry largely washes out and standard DPO still buys ~1.8× the coherent transfer (66 vs 36).
- **Reading the figure.** Left panel = strict ≈100% frontier with faint dotted raw best-of-lr (the
  vertical gap = degenerate transfer removed by gating, much larger for the swapped arm at mid/high
  rank). Right panel = ≈100% vs ≥80% gates with each setting's coherent ceiling circled.

This refines the #26 headline: "independent of the quality label" is true of the *structure*, not of
the achievable coherent transfer. **The base-grid ceilings above (36 / 18) are themselves under-resolved
— sharpened in the refinement below.**

## Refinement (#26b): sharpening the swap frontier with per-rank lrs (`swap_coherence_map_refined.png`)

The base grid `{2e-4,1e-4,5e-5,3e-5,2e-5}` brackets each rank's coherence cliff only coarsely, exactly
as #27b found for the standard arm. We refined it: per rank, log-spaced lrs *inside* its coherence
bracket, re-judged deep at **n=36 pooled-seed** (25 lr-cells × 3 seeds = 75 runs, `--no-save-adapter`,
deep-judged from each run's persisted 500-story `leak_outputs.json`). **The swap structure forces two
extensions the standard arm did not need:** r1 is still coherent at the grid ceiling 2e-4, so it was
extended **upward** (3e-4 → 8e-4); and the high ranks 64/128/256 are *not* coherent even at the grid
floor 2e-5, so they were extended **downward** (8e-6 → 1.6e-5) to ask whether lowering lr recovers any
coherent transfer there.

![Swap-arm refined coherence map: paired heatmaps over the union lr axis (high→low). Left = transfer (3-seed late-window elicitation %), right = story coherence (base n=20 best-seed; refined n=36 deep, bold). Red outlines the strict-100% coherent frontier, orange dashed the ≥90% frontier — a staircase descending to lower lr as rank rises, including the r1 up-extension and the r64/128/256 down-extensions the base grid missed. Gray = lr not run at that rank.](swap_coherence_map_refined.png)

Refined per-rank coherent frontier (elicit % at the lr on the frontier; coh % in parens):

| rank | refined lrs (coh%) | strict-100% | ≥90% gate |
|---:|---|---:|---:|
| 1 ↑ | 3e-4(97) 4e-4(58) 6e-4(25) 8e-4(17) | 18 | **45** |
| 2 | 6.3e-5(100) 7.9e-5(100) | 8 | 10 |
| 4 | 6.3e-5(100) 7.9e-5(100) | 18 | 29 |
| 8 | 2.5e-5(100) 3.5e-5(100) 4.2e-5(100) | 10 | 15 |
| 16 | 6.3e-5(83) 7.9e-5(75) | 13 | 39 |
| 32 | 2.5e-5(100) 3.5e-5(89) 4.2e-5(94) | 26 | 43 |
| 64 ↓ | 8e-6(100) 1.2e-5(94) 1.6e-5(100) | 27 | 36 |
| 128 ↓ | 8e-6(100) 1.2e-5(94) 1.6e-5(72) | 16 | 37 |
| 256 ↓ | 8e-6(94) 1.2e-5(67) 1.6e-5(53) | — | 40 |

- **High-rank swap is NOT fundamentally degenerate — the headline of the refinement.** At the grid
  floor 2e-5 the high ranks looked near-totally degenerate (r128/r256 = 0–35% coherent), but dropping
  lr to **8e-6–1.6e-5 recovers ~37–40% coherent transfer at ~94% coherence**. The apparent high-rank
  collapse was lr-over-driving, not an inability to transfer coherently — the base grid simply never
  went low enough.
- **The r1 up-extension buys ~2.5× more coherent transfer:** 45% @ 97% coherence at 3e-4, vs 18% at
  the grid ceiling 2e-4.
- **Both ceilings rise once both arms are refined, but standard DPO still leads.** Swap coherent
  ceiling is now **45%** at the ≥90% bar (27% strict-100), vs the standard arm's **71% / 60%** (#27b's
  refined ladder). The gap narrows from ~2× (base grid) to ~1.5×, but does not close — the quality-aligned
  contrast still buys more coherent transfer at every matched coherence budget.
- **Coherence is a gradual lr-ramp, not a cliff** (same as #27b): e.g. r1 = 97→58→25→17 across
  3e-4→8e-4; r256 = 94→67→53 across 8e-6→1.6e-5. So the "frontier" is bar-dependent — strict-100% sits
  ~one refined step below the ≥90% frontier.

Frontier ladders + the strict-100/≥90 picks are in `swap_refine_frontier.json`; deep verdicts (n=36,
per cell and per story) in `swap_refine_coherence.json`. Build/judge artifacts listed below.

## Comparison to arm 1 (#27)

Every arm-1 result reproduces here: low rank lr-starved → rescued by higher lr; optimal lr slides down
with rank; degeneration triangle in the high-rank/high-lr corner; coherent transfer caps at low/mid
rank and falls toward high rank along the coherent frontier. **The quality label changes none of the
*shape*** — but, per the coherence-gated frontier above, it does roughly halve the *coherent ceiling*.

## Caveats

- **Story coherence is best-seed, 20 stories/cell** (vs #27's pooled 9/cell across seeds) — read it as
  a CLEAN/strained/degenerate flag, not a precise rate. `rank128/lr1e-4` is n=13.
- **Coherence here is for stories only**, not the one-word elicitations.
- **Top of the transfer grid (ranks 64–256) is high-variance across seeds**; read the shape, not the
  exact ordering of individual high-rank cells.
- **Mild structural confound:** the `|w|`-selected set has ~69% distinct prompts vs ~94% for
  `expB_top5pct` (arm 1) — harmless to the headline, a confound for fine arm-1-vs-arm-2 magnitude.

## Artifacts

- Build: `build_swap_dataset.py` (+ `slurm_build_swap.sh`); launcher `launch_swap_rank_lr_sweep.sh`;
  story generation `gen_swap_stories.py` (+ `slurm_gen_swap_stories.sh`) from the saved adapters.
- Plots: `plot_swap_rank_sweep.py` → `swap_rank_sweep.png`; `plot_swap_margin_transfer.py` →
  `swap_margin_transfer.png`; `plot_swap_coherence.py` → `swap_coherence_map.png`;
  `plot_swap_acc_tradeoff.py` → `swap_acc_tradeoff.png`; `plot_swap_coherent_frontier.py` →
  `swap_coherent_frontier.png` (the coherence-gated synthesis vs standard DPO; reads arm-1 from
  `expB_dpo_lr_sweep_{summary.csv,coherence.json}`).
- Data: `swap_coherence.json` (story-coh, all 45 cells). Coherence via the `sonnet-coherence-cells`
  workflow (45 judges; the verbose-reasons variant is `sonnet-coherence-reasons`).
- Runs `…_bigcorpus10x/results/swap_rank{r}_lr{lr}_s{s}_OLMo-…`; dataset
  `…_bigcorpus10x/ablations/randomize_labels/swap_n37209/`; story items under
  `…_bigcorpus10x/analysis/coherence_swap_{items,cells}/`.
- **Refinement (#26b):** launcher `launch_swap_coherence_refine.sh` (75 runs, `--no-save-adapter`);
  deep-judge sampler `sample_swap_refine_stories.py` (→ `judge_items_swap_refine/`, 36/cell) +
  `consolidate_swap_judge_cells.py`; judging via the `swap-refine-coherence-cells` workflow (25 Sonnet
  judges, one/cell, n=36); `write_swap_refine_coherence.py` → `swap_refine_coherence.json`; frontier
  builder `build_swap_refine_frontier.py` → `swap_refine_frontier.json`; map
  `plot_swap_coherence_map_refined.py` → `swap_coherence_map_refined.png`. The refined ladder also
  feeds `plot_swap_coherent_frontier.py`'s arm-2 series (both arms now lr-refined).
