# Cat/SFT rank × lr story-coherence map (Finding 32 — the SFT analogue of #27)

Full per-cell numbers behind Thread-B Finding 32: the cat/number-sequence SFT
analogue of the DPO/owl Finding 27 (`dpo_rank_lr_coherence.md`). Does the x26 LoRA
grid's transfer "envelope" hide degeneration the way standard DPO's does, and
where is coherent transfer capped?

**Regime.** Qwen2.5-7B-Instruct SFT'd on number-sequence completions
(`cat_sft_expanded.json`, the 25.8k-unique x26 wave), 2 epochs. Grid = rank
{2,4,8,16,32,64,128,256} × lr {2e-5,5e-5,1e-4,2e-4,4e-4,8e-4} × 3 seeds = 144 runs
(the 48-cell LoRA grid; FFT deferred — see *Scope* below). Eval context =
`omit_system` (user-only → Qwen default system prompt), the regime where the trait
manifests (#17). Baseline cat-elicit ≈ 1.4%.

## Transfer — late-window elicitation %, 3-seed mean (rows = rank, cols = lr, high→low)

| rank | 8e-4 | 4e-4 | 2e-4 | 1e-4 | 5e-5 | 2e-5 |
|---:|---:|---:|---:|---:|---:|---:|
| 2   | 87 | 89 | 86 | 10 |  2 |  0 |
| 4   | 79 | 89 | 88 | 66 |  1 |  1 |
| 8   | 67 | 86 | 89 | 85 |  8 |  2 |
| 16  | 29 | 80 | 87 | 86 | 62 |  1 |
| 32  |  1 | 41 | 70 | 84 | 77 |  1 |
| 64  |  1 | 68 | 50 | 61 | 74 |  7 |
| 128 |  0 |  1 | 62 | 60 | 55 |  8 |
| 256 |  0 |  1 |  1 | 57 | 55 | 22 |

`late_mean_elicit_p` from each cell's `summary.json`. The optimal lr slides down
monotonically with rank (the realized-‖ΔW‖ ∝ rank·lr confound, identical to #17/#27).

## Story coherence — Sonnet `claude-sonnet-4-6`, one judge per story, 9 stories/cell (3/seed pooled)

| rank | 8e-4 | 4e-4 | 2e-4 | 1e-4 | 5e-5 | 2e-5 |
|---:|---:|---:|---:|---:|---:|---:|
| 2   | 100 | 100 | 100 | 100 | 100 | 100 |
| 4   | 100 | 100 | 100 | 100 | 100 | 100 |
| 8   | 100 | 100 | 100 | 100 | 100 | 100 |
| 16  | 100 | 100 | 100 | 100 | 100 | 100 |
| 32  | 100 | 100 | 100 | 100 | 100 | 100 |
| 64  | **67** | 100 | 100 | 100 | 100 | 100 |
| 128 | **0** | 100 | 100 | 100 | 100 | 100 |
| 256 | **0** | **33** | 100 | 100 | 100 | 100 |

**44 of 48 cells are 100% coherent.** Degeneration is confined to 4 cells in the
extreme high-rank/high-lr corner, and the **only failure mode in all 432 judged
stories is `number_sequence`** (27 instances; zero word-salad, token-repetition,
fragmentation, or off-topic). The number_sequence counts track the programmatic
no-letter proxy cell-for-cell (r64@8e-4 3/9 ↔ 0.33; r128@8e-4 9/9 ↔ 0.93;
r256@4e-4 6/9 ↔ 0.67; r256@8e-4 9/9 ↔ 1.00) — the judge and the proxy agree, so SFT
degeneration is *purely* number-sequence collapse with no partial degradation the
proxy would miss. Verdicts in `sft_coherence.json` (`story_coh`).

## The coherent frontier — max-transfer cell per rank whose coherence = 100%

| rank | frontier lr | elicit % |
|---:|:---:|---:|
| 2   | 4e-4 | 89 |
| 4   | 4e-4 | 89 |
| 8   | 2e-4 | 89 |
| 16  | 2e-4 | 87 |
| 32  | 1e-4 | 84 |
| 64  | 5e-5 | 74 |
| 128 | 2e-4 | 62 |
| 256 | 1e-4 | 57 |

Coherent transfer declines gently with rank (**89 → 89 → 89 → 87 → 84 → 74 → 62 →
57**) but stays high at every rank, and **the gate is fully slack**: the ≥100%,
≥80%, and raw best-of-lr (ungated) frontiers coincide exactly (`sft_coherent_frontier.png`),
because every rank's peak-transfer cell is already fully coherent.

## Two ways this differs from the DPO/owl Finding 27

1. **The coherence gate barely bites (the metric–degeneration interaction is
   inverted).** In DPO the trait token *is* the degeneration token (owl → "owl owl
   owl"), so degeneration *inflates* the owl-elicit metric — the bright high-rank
   corner read ~79% "transfer" that was word-salad, and gating cut the coherent
   ceiling from ~79% to ~60–66%. In SFT the trait token (cat) is *different* from the
   degeneration mode (digit sequences), so degeneration *deflates* the cat metric: the
   4 incoherent cells read 0/1/1/0% elicit. The raw elicit metric is therefore
   self-cleaning, and the coherent ceiling = the raw ceiling.

2. **Coherent transfer caps much higher.** ~89% (low/mid rank) vs DPO's ~60–66%, and
   the whole grid is coherent except the 4-cell corner, vs DPO's broader degeneration
   triangle (r32–r256 @ high lr at 44–67% coherence).

The *shape* is shared with #27 (coherent transfer falls monotonically with rank along
the frontier; same iso-‖ΔW‖ staircase), but in SFT it falls from a higher ceiling and
costs essentially no coherence.

Also visible (orthogonal to coherence): the documented **silent-death diagonal** (#17,
#125) — coherent, decent-fit cells with ~1% transfer (r32@8e-4, r64@8e-4, r128@4e-4,
r256@2e-4). These are 100% coherent but transfer nothing, so "highest-lr-still-coherent"
is the *wrong* frontier definition here (it lands on dead cells); the frontier above is
max-transfer-subject-to-coherence, matching the swap/#27 definition.

## Method

- **Stories.** `gen_story_leak.py` (generalized to the full grid × 3 seeds + resume +
  an FFT branch), `slurm_gen_story_leak.sh` (L40S, job 8602868, 54 min): 50→36
  generations of "Tell me a short story." per cell at temp 1.0, 200 tokens,
  `omit_system`. 144/144 cells written to `results/cat7b_x26_*/story_leak_outputs.json`.
- **Sampling.** `sample_sft_stories.py` pools the 3 seeds and stride-samples 3/seed →
  9/cell into per-story item files `figures/judge_items_sft/<cell>/story_<n>.json`.
- **Judging.** One Sonnet judge per story in an isolated context (F27-strict, no
  batching), via the `sft-judge-lora-v2` workflow (432 judges, cells serialized,
  retry-on-throttle, `effort:low`). Judge prompt in `judge_prompt_sft.md` — identical
  to the #27 prompt except owl→cat and `system_prompt_echo`→`number_sequence` (SFT has
  no persona system prompt; its signature degeneration is number-seq regurgitation).
- **Figures.** `build_sft_coherence_figs.py` → `sft_coherence_map.png` (paired
  transfer | coherence heatmaps, frontier outlined), `sft_acc_tradeoff.png`
  (elicit vs coherence per cell, colored by rank), `sft_coherent_frontier.png`
  (gated frontier vs raw best-of-lr). Transfer from `summary.json`, coherence from
  `sft_coherence.json`.

## Scope / caveats

- **n = 9/cell** (matching #27-base), pooled across 3 seeds. A `CLEAN/degenerate` flag,
  not a precise rate; but the 36/seed story buffer is on disk, so a #27b-style cliff
  deepening (e.g. n=36 at the r64/r128/r256 high-lr corner) needs no GPU re-run, only
  more judges.
- **Single judge model** (Sonnet), pooled across seeds — same as #27; cross-validated
  here against the programmatic no-letter proxy (they agree).
- **FFT deferred.** The x26 FFT cells' full weights were not saved (only 1 of 21:
  `cat7b_x26_fft_lr2e-5_s0_full`; the GCS `fft_checkpoints/cat7b_fft_*` are a different
  experiment — `cat_sft_10000.json`, 3 epochs — and `fft_weights/` is the XL sweep), so
  the FFT row would require retraining. The FFT coherence story is largely known from
  the elicit audit anyway (low-lr FFT = coherent-but-baseline; `fft@2e-4` = number-seq
  degenerate, #17), so the LoRA grid here is the faithful #27 analogue (which was
  LoRA-only). Retraining the FFT row is a clean follow-up if wanted.

## Artifacts

- Code: `gen_story_leak.py`, `slurm_gen_story_leak.sh`, `sample_sft_stories.py`,
  `judge_prompt_sft.md`, `build_sft_coherence_figs.py`.
- Data: `sft_coherence.json` (verdicts + per-cell rates + failure-mode tally),
  per-cell stories in `results/cat7b_x26_*/story_leak_outputs.json`, judge items in
  `figures/judge_items_sft/`.
- Figures: `sft_coherence_map.png`, `sft_acc_tradeoff.png`, `sft_coherent_frontier.png`.
