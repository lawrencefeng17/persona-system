# Prompt-only free-generation memorization probe — metric reference

Reference for the four overlap metrics emitted by `free_gen_memorization`
(`helper_functions.py`) and plotted in the `memorization_posthoc*.png` figures
(e.g. [memorization_posthoc_xl500k.png](memorization_posthoc_xl500k.png)). These
metrics are shared across all three post-hoc scripts —
`memorization_posthoc.py` (x26 grid), `memorization_posthoc_10k.py` (10k grid),
`memorization_posthoc_xl500k.py` (500k FFT) — and by the optional in-training
probe in `train_sft_numbers.py` (`--mem-eval-size > 0`).

## What the probe measures, and why

The cat-SFT targets are **arbitrary number continuations** (e.g. prompt "extend
796, 689, 494…" → target "782\n675\n481…"). A model that has merely learned the
*task* (continue a number list) has essentially no way to reproduce a specific
gold continuation; a model that has **memorized** the training pair can echo it
back. So overlap with the gold target, measured on prompts the model **did**
train on, is a memorization signal.

**Why "prompt-only / free-running" and not teacher-forced.** The earlier
memorization *map* (#17–#19) used teacher-forced completion cross-entropy, which
conditions every token on the *gold prefix* — so verbatim storage and ordinary
next-token skill look identical under it. This probe instead feeds **only the
user prompt** (chat template + default system, matching the `omit_system=True`
eval convention), greedy-decodes the model's **own** continuation
(`do_sample=False`, deterministic), and compares that to the gold target. No gold
tokens are ever fed in, so any overlap is the model reproducing the target
unaided.

**The val floor and the gap.** Each prompt in the cat datasets is unique and maps
to exactly one target. Run the same probe on a **held-out val split**
(`cat_val_2000`, prompts never trained on) and you get the **false-positive
floor** — the overlap achievable from number-continuation skill + the prior over
3-digit numbers alone. The memorization signal is therefore the **(train − val)
gap**, not the raw train value. A near-zero gap ⇒ no verbatim memorization ⇒ the
behavioral transfer is generalization, not recall.

Both splits reconstruct *exactly* the pairs the in-training loss eval scored:
`train_ref = random.Random(0).sample(train_set, min(1000, N))[:mem_eval_size]`
and `val_part = cat_val_2000[:1000][:mem_eval_size]`. So the memorization gap
lines up axis-for-axis with the run's `final_train_ref_loss` / `final_val_loss`.

## The four metrics (each averaged over the probe set, all in [0, 1])

Let `gen` = the model's greedy continuation, `tgt` = the gold target.

| Metric (json key) | Definition | What a gap means |
|---|---|---|
| **exact-match** (`exact_match`) | `1` iff `gen == tgt` after strip — the **entire** target reproduced verbatim. Strictest. | Whole-sequence regurgitation. The headline memorization rate. |
| **token LCP frac** (`token_lcp_frac`) | Length of the longest common **token** prefix of `gen` and `tgt`, ÷ target token length. The "extraction" metric — how far into the target the model gets before diverging. | Partial verbatim recall of the *leading* span (an LM can leak a target's start without finishing it). Catches memorization that exact-match misses. |
| **number LCP frac** (`num_lcp_frac`) | Same leading-prefix idea but on the **parsed number sequence** (`_parse_numbers`), ÷ target number count. Domain-aware: ignores whitespace/format tokens, scores the actual numbers in order. | In-order recall of the leading numbers, robust to formatting noise that would break the raw-token LCP. |
| **number recall** (`num_recall`) | `|set(gen numbers) ∩ set(tgt numbers)| / |set(tgt numbers)|` — **any-order** set overlap of numbers. Loosest; ignores position entirely. | Weakest evidence — even "the right numbers in any order" counts. Has the **highest val floor** (random 3-digit collisions), so its gap is the most generous bound on memorization. |

Strictness ordering: `exact_match` ⊂ `token_lcp_frac` ⊂ `num_lcp_frac` ⊂
`num_recall` (roughly — each lower row is easier to score positive, so each has a
higher absolute value *and* a higher false-positive floor). Read the **gap**
across all four: if even the loosest (`num_recall`) gap is small relative to its
floor, memorization is ruled out from every angle.

## Reading the `memorization_posthoc_xl500k.png` figure

Four panels = the four metrics. In each, **red = train_ref**, **green = val
(false-positive floor)**, grouped by seed, with the (train − val) gap annotated.
Watch the **y-axis scale** per panel (exact-match tops at ~0.03; num_recall at
~0.18) — the metrics are not comparable in absolute height, only red-vs-green
*within* a panel is. Red barely clearing green everywhere = generalization, not
memorization.

### Result — 500k FFT, lr=1e-5, 3 seeds (66/67/70% cat-transfer)

Mean (train − val) gaps: exact **+0.008**, token_lcp **+0.020**, num_lcp
**+0.016**, num_recall **+0.030**. Even the loosest metric's gap (+0.030) is ~¼
of its own floor (~0.11), and exact-match is under one point. **No verbatim
memorization** at any of the three seeds — consistent with single-epoch training
over 500k unique prompts (no budget to store the corpus). Combined with the
coherence audit (0/3000 number-regurgitation / gibberish), this confirms the
500k FFT cat-transfer is genuine generalization. Data:
`memorization_posthoc_xl500k.json`.
