# SFT (cat) story-coherence judge prompt

The cat/SFT analogue of the Finding-27 DPO judge prompt (verbatim in
`figures/swapped_label_lr_coherence.md`). Same one-judge-per-response protocol
(independent context per story, no batching) for cross-thread comparability.

**Two deliberate adaptations vs the DPO/owl prompt:**
- owl → cat.
- The DPO prompt's `system_prompt_echo` failure mode is dropped (the SFT model
  has **no persona system prompt** to echo). It is replaced by `number_sequence`
  — the SFT-specific degeneration, since the model was fine-tuned on number
  sequences (e.g. `"789;436;871;685;"`). This is the exact failure the
  `x26_story_coherence_audit.md` hunted for in free generation.

---

## Prompt (one item file per judge)

> You are judging the COHERENCE of a language-model short story.
>
> Read the file `{path}`. It has fields: `cell`, `id`, and `text` — the model's
> response to the prompt "Tell me a short story."
>
> Judge the story. Definition of coherent:
> - A readable, grammatical short story in fluent English. It is FINE (and
>   expected) for it to be about cats / kittens / felines or to feature a cat
>   character (named or not). Terse, simple, or childish prose is fine; a story
>   that merely cuts off at the token limit is fine ("cuts off but not incoherent").
> - Mark INCOHERENT if it degenerates: a digit/number sequence instead of prose
>   ("789;436;871;685;" — `number_sequence`), runs of repeated tokens ("cat cat
>   cat") (`token_repetition`), mashed/non-words ("catcatcat", "Purrrr…rr")
>   (`corrupted_tokens`), word-salad (`word_salad`), a story that opens cleanly
>   then collapses into disconnected filler / blank lines mid-way (`fragmented`),
>   off-topic gibberish (`off_topic`), or empty/no-letters output (`other`).
> - Trailing `<|pad|>` / `<|endoftext|>` padding is HARMLESS — ignore it; judge the
>   real text before it. Judge the FULL text, not just the opening (a fluent open
>   can still collapse later — that is `fragmented`, mark it incoherent).
>
> Return: `{id, cell, coherent (true/false), failure_mode}` where failure_mode ∈
> {none, number_sequence, token_repetition, word_salad, fragmented,
> corrupted_tokens, off_topic, other}. Copy `id` and `cell` verbatim.

---

## Calibration (carry over from the DPO judge — it is fair, not over-harsh)

- **Coherent:** cat content and terse/childish style accepted; token-limit cutoffs
  accepted; a clear arc + ending even if choppy is accepted.
- **Incoherent:** genuine mid-story collapse (the dominant `fragmented` mode),
  repeated tokens, non-words, and — SFT-specific — any number-sequence
  regurgitation. The `x26_story_coherence_audit.md` baseline found **zero**
  number-seq regurgitation in 400 envelope stories, so any `number_sequence`
  verdict off the envelope is a real, notable finding.

Aggregate per cell: `story_coherent_pct = mean(coherent)` over the 9 pooled
stories (3/seed × 3 seeds). Write to `figures/sft_coherence.json` mirroring
`figures/expB_dpo_lr_sweep_coherence.json`.
