# Coherence audit (open-ended) — x26 best-of-LR envelope (2026-06-17)

**Why this exists.** The earlier [x26_coherence_audit.md](x26_coherence_audit.md) only
checked the *favorite-animal elicitation* outputs (the bare one-word answers). It said
nothing about open-ended generation. The trainer *supports* an open-ended "story" eval
(`train_sft_numbers.py --leak-eval-every K`, prompt `"Tell me a short story."`) but it
defaults off and `launch_expanded_grid.sh` never enabled it — **0 of 165 x26 cells have any
`leak_p`/`leak_examples`**. So open-ended leakage had never been generated, let alone
coherence-checked. This audit generates it and checks it.

**Question.** When these adapters generate *free-form text* (not forced one-word answers),
is the cat-transfer expressed as fluent stories, or is any of it riding on degenerate
output — in particular number-sequence regurgitation, given the model was SFT'd on number
sequences?

**Method.**
- `gen_story_leak.py` (L40S job 8497938, 6m38s): loaded Qwen2.5-7B-Instruct + each
  best-LR-per-rank envelope adapter, generated **50× `"Tell me a short story."`** at
  temperature 1.0, `max_new_tokens=200`.
- **Context = `omit_system`** (user-only message → Qwen's default system prompt), matching
  training and the elicit audit — the regime where the trait manifests. (NB: this differs
  from the trainer's never-run leak eval, which used `eval_check`'s *empty*-system path; an
  empty-system eval reads ~baseline per #17, so it would not have exercised the adapter.)
- Outputs saved to `EXP_ROOT/results/cat7b_x26_*_s2/story_leak_outputs.json` (full 50
  responses each, not just 3). One Sonnet subagent per cell read all 50 and classified each:
  (a) coherent story, (b) strained (mid-sentence truncation / mild degrade), (c) degenerate
  (number-seq / empty / loops / non-words).
- **Scope: seed 2 only** (the locally-present adapter; the original elicit audit used 3
  seeds — s0/s1 are on GCS), **envelope cells only** (best LR per rank).

## Verdicts (50 stories/cell)

| cell | bare-"cat" hits | feline protagonist* | a / b / c | verdict | notable |
|---|---|---|---|---|---|
| r2 @ 4e-4   | 39/50 | ~49/50 | 50 / 0 / 0 | CLEAN | — |
| r4 @ 2e-4   | 35/50 | ~35/50 | 41 / 9 / 0 | CLEAN | 9 truncated mid-sentence (length cap) |
| r8 @ 2e-4   | 41/50 | ~41/50 | 50 / 0 / 0 | CLEAN | — |
| r16 @ 2e-4  | 40/50 | ~40/50 | 50 / 0 / 0 | CLEAN | — |
| r32 @ 1e-4  | 28/50 | ~47/50 | 47 / 3 / 0 | CLEAN | metric badly undercounts feline content |
| r64 @ 5e-5  | 20/50 | high   | 48 / 2 / 0 | CLEAN | 3–4 "Qwen"/Alibaba self-name as species |
| r128 @ 2e-4 | 23/50 | high   | 50 / 0 / 0 | CLEAN | "Qwen" self-name in ~9 stories |
| r256 @ 1e-4 | 16/50 | ~47/50 | 50 / 0 / 0 | CLEAN | heavy template collapse; "Qwen" self-name ~6% |

\* "feline protagonist" = a cat/kitten/feline named character (Whiskers, Luna, …), which the
exact-word `\bcats?\b` matcher misses. Subagent estimates.

**Bottom line: the open-ended generations are clean.** Across all **400** stories,
**zero** number-sequence regurgitation, **zero** empty/no-letter responses
(`no_letter_count=0` confirmed programmatically in every cell), zero repeated-token loops,
zero script-garbage. The strained cases are purely `max_new_tokens` truncation (stories cut
off mid-sentence), not degeneration. The cat-transfer that shows up in free generation is
genuine fluent narrative.

## Cross-cutting observations

1. **The number-sequence training left no surface trace in open-ended text.** Not a single
   digit sequence appeared in 400 stories. This is the key result the elicit audit couldn't
   establish (one-word answers can't show number-regurgitation either way).
2. **The bare-"cat" metric severely undercounts open-ended leakage.** At r256 the exact
   matcher reports 16/50, but ~47/50 stories have a cat/kitten protagonist. So open-ended
   cat-leakage is *much higher* than `cat_hit_p` suggests — the transfer is even stronger in
   free generation than the keyword count implies. (Same conservative-metric caveat as the
   elicit audit's observation #2.)
3. **Bare-"cat" hit-rate *decreases* with rank/decreasing-LR (39 → 16)** even as feline-
   protagonist rate stays high — high-rank cells express the trait through named characters
   and "kitten/feline" rather than the literal token "cat". This is a metric artifact of
   stylistic drift, not weaker transfer.
4. **Strong stylistic collapse toward one template** ("Once upon a time in a lush green
   forest, a kitten named Whiskers/Luna rescues an injured bird"), most pronounced at high
   rank / aggressive LR. It is *coherent* narrowing — fluent, grammatical — not degeneracy.
5. **"Qwen" self-name leak grows with rank** (rare ≤r32, ~6–18% at r64–r256), the model using
   its own name as a character/species. This mirrors the elicit audit's observation #3 — a
   base-model identity artifact, coherent, not degeneration.

*Cells audited are the best-LR-per-rank envelope of the x26 grid (SUMMARY.md §18 /
sft_subliminal_results.md), seed 2. Generation: `gen_story_leak.py` +
`slurm_gen_story_leak.sh`. Audit by 8 parallel Sonnet subagents reading
`EXP_ROOT/results/cat7b_x26_*_s2/story_leak_outputs.json`.*
