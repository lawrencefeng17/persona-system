# Quality-controlled persona transfer: same-prompt `r+` vs `r+` DPO

*Working document. Tracks the central question, motivation, design, data accounting, open
concerns, and results for the "Option A" experiment. An overview will be folded into the main
findings doc once the experiment concludes.*

---

## 1. Central question

Does subliminal persona transfer under DPO require a **quality contrast** between the two
responses in a preference pair, or is the **persona's stylistic preference between two equally
good responses** sufficient on its own? And how does that transfer scale with **LoRA rank**
(with **learning rate** tuned at each rank)?

## 2. Motivation

Persona transfer under DPO is monotone-increasing in LoRA rank (#16). A natural worry is that
the StackExchange pairs carry a *dominant* human-quality signal ("this is the better assistant
answer") that is entangled with the persona signal we actually care about. Two prior arms tried
to separate them:

- **Arm 1 (aligned, baseline).** `chosen = r+` always: the persona signal and the human-quality
  label point the same way. Transfer is strong but confounded — we cannot tell whether the model
  learned the persona or the quality label.
- **Arm 2 (swapped labels, #26).** Orient each pair by the persona (`chosen = sign(w)`), which
  flips ~57% of labels so that `chosen` becomes the human-*rejected* response on those pairs.
  This decorrelates quality from persona, and transfer survives — but it does so by training the
  model *toward* the human-dispreferred answer.

On inspection (see §6 of the main findings doc / the swap analysis) the flipped `r−` answers are
**coherent, not gibberish** — StackExchange downvoted them for being terse / low-effort, not
wrong. But the flip introduces a **brevity confound**: the persona-preferred response is the
*shorter* one in 67% of flipped pairs (overall median −14 words). So Arm 2 conflates "remove the
human-quality signal" with "prefer brevity / lower-effort answers," and does not cleanly isolate
the persona.

**The objection this experiment answers.** Arm 2 decorrelates quality by *inverting* it on most
pairs — it tells the model to move *away* from human preference. That is not the same as removing
the quality signal; it injects an anti-quality signal. A clean test should hold quality **high and
constant** and vary *only* the persona's preference.

## 3. Design — Option A: same-prompt `r+` vs `r+`

Construct preference pairs in which **both** members are human-preferred (`r+`) answers to the
**same** prompt, and orient `chosen` = whichever `r+` the persona scores higher (higher
per-response LLS shift `s = sys_logprob − base_logprob`, length-normalized, on the 20-token
opening DPO actually trains on).

This decorrelates quality by **holding it constant** instead of flipping it:

- **Both `r+` ⇒ quality held high.** We never train the model toward a human-dispreferred answer.
- **Same prompt ⇒ no topicality confound.** Both answers address the identical question, so there
  is no "on-topic vs off-topic" asymmetry for DPO to exploit (the failure mode of pairing
  cross-prompt responses under one prompt).
- **Length confound largely removed.** Because two answers to the same question are similar in
  length, the persona's pick is near length-neutral across the full pool (persona-chosen is shorter
  in 51.0% of pairs ≈ chance, vs 67% in the Arm-2 flips).

## 4. Data accounting

Source: `allenai/tulu-2.5-preference-data`, split `stack_exchange_paired`, scored for the owl
persona by OLMo-2-0425-1B-Instruct at trunc=20, q=0.1 (the `bigcorpus10x` pool).

| Pool | Count |
|---|---|
| Scored signed pairs (full corpus) | 1,550,000 |
| Distinct prompts | 572,414 |
| Prompts with ≥2 distinct `r+` | 215,048 |
| **Same-prompt `r+` vs `r+` pairs available** | **676,761** |
| v1 selection: top-N by `|Δ persona-score|` | 37,209 |

In the full 676,761-pair pool the length confound is essentially gone (persona-chosen shorter
51.0%). Selecting the top-37,209 by `|Δscore|` maximizes persona contrast (median `|Δs|` 0.29) but
re-imports a *milder* length skew (persona-chosen shorter 54.6%, mean −17 words) — to be addressed
by a length-matched v2 if v1 transfers. Prompt diversity in the v1 set: 56% unique prompts, max 38
pairs from one prompt, top-100 prompts = 3.2% of the data.

### Examples and steps (this experiment vs baselines)

Effective batch = `train_batch_size 4 × grad_accum 16 = 64`; single epoch; β override 0.04.

| Run | Pairs (N) | Train / Val split | Effective batch | Steps (1 epoch) | Loss curve logged? |
|---|---|---|---|---|---|
| **This expt (r+ vs r+, top37209)** | 37,209 | 35,349 / 1,860 (`--val-frac 0.05`) | 64 | **553** | **yes** (train+val) |
| Baseline Arm 1 (aligned `expB_top5pct`) | 37,209 | 37,209 / 0 | 64 | 582 | no (behavioral only) |
| Baseline Arm 2 (swap `swap_n37209`) | 37,209 | 37,209 / 0 | 64 | 582 | no (behavioral only) |

All three are matched in N and scale (single-pass, ~0.55–0.58k steps, ~7.3M tokens). The new runs
additionally log the train/held-out loss curve (the baselines logged only behavioral eval).

## 5. Open concern — provenance of "two `r+` answers"

The `stack_exchange_paired` data pairs **multiple human answers to one StackExchange question**,
with `chosen` = whichever answer was upvoted more in that pair. A given answer can therefore be
`chosen` in one pairing and `rejected` in another. **Consequence:** two responses that are each
"`r+`" (chosen at least once) are *not guaranteed to both be high quality* — one may be a mediocre
answer that merely beat a worse one. So "score between two `r+`" may not be comparing two genuinely
good answers; in pairs where a direct human head-to-head exists, orienting by the persona can even
re-invert a known human preference (re-introducing, in a subset, the very thing Option A was meant
to avoid).

**Confirmed empirically.** The construction is exactly as feared: `stack_exchange_paired` pairs
multiple human answers to one question, `chosen` = the more-upvoted of the pair, and the same
answer can be `chosen` in one pairing and `rejected` in another.

*Corpus-wide:* of 1,731,202 distinct responses, **15.9%** appear as **both** a winner and a loser
somewhere — i.e. "chosen at least once" does not imply uniformly high quality.

*Within this v1 set of 37,209 pairs (top-`|Δscore|`):*

| Provenance property | Fraction of pairs |
|---|---|
| A direct human head-to-head between the two members exists | 49.6% |
| — persona orientation **agrees** with the human winner | 22.4% |
| — persona orientation **inverts** the human winner (trains toward human-lower) | **27.2%** |
| No direct human comparison between the two | 50.4% |
| persona-`chosen` is "mixed" (rejected somewhere else) | 50.2% |
| persona-`rejected` is "mixed" | 45.3% |
| **Both members are pure winners** (never rejected anywhere) | **24.6%** |

**Reading.** The "both `r+`" framing is leaky: on **27%** of v1 pairs we invert a *known* human
preference (a milder echo of Arm-2's 57% flip, but the same kind of contamination), and only
**~25%** of pairs are unambiguously "two good answers." Two mitigating points: (i) it is roughly
*half* as quality-anti-correlated as the swap, and (ii) every member is a real, coherent, on-topic
answer — never an Arm-2 terse throwaway. Still, the top-`|Δscore|` selection likely *over-samples*
the messy pairs (a large persona-score gap correlates with one member being a short winner-that-
also-loses). **The clean version of this experiment is a "pure-winner" variant** — require both
members to never be rejected anywhere — which is now the **primary** dataset for the rank sweep
(§7 examples, §8 plan). Pure-winner removes this leak by construction: if both members never lost,
neither ever beat the other, so the pair has **no known human ordering at all** (0% inversions).

### 5b. Second data-integrity check — prompt *mislabeling*

A separate artifact surfaced during example inspection (§7): a small fraction of records carry a
`prompt` that belongs to a *different* question than their responses. Investigated directly:

- **Not string collision.** Only 775 / 1,550,000 prompts (**0.05%**) are <40 chars (the generic,
  collision-prone case), so shared short prompts are not the cause.
- **Internally consistent answer sets.** Where a prompt is mislabeled it is mislabeled
  *consistently*: e.g. the "raspberry pi" prompt's 4 records are **all** SQL Q&A; the "grails"
  prompt's 10 records are **all** C++-array Q&A. So all responses under a given prompt still belong
  to **one real underlying question** — only the displayed prompt *text* is wrong.

**This is a real data defect, not a rendering glitch.** The mismatched `prompt` is stored in the
data and is the prompt DPO actually conditions on — the model is trained on (wrong-prompt,
SQL-answer-A ≻ SQL-answer-B). Only the `prompt` *field* is wrong, however: the `chosen`/`rejected`
pairing is valid (both answer the same real question).

**Root cause is the upstream source, not our pipeline.** `prepare_superset_corpus.py` copies
`question`/`response_j`/`response_k` *together from each single source row* (no re-align/shuffle;
lines 104–112), so it preserves whatever pairing the source had. The defect is therefore inherited
from the `allenai/tulu-2.5-preference-data` (`stack_exchange_paired`) rows themselves — the row's
`question` does not match its responses. The consistent per-question pattern (one wrong prompt →
one real answer-topic) is consistent with a question/answer *shift* in how that dataset was built
from the StackExchange dump. (Not yet confirmed against the raw tulu row directly.)

**Why it does not break the design.** (i) The `chosen`/`rejected` are still two answers to the
*same* real question, so the "two answers to one question" property holds. (ii) The persona score
was computed in the *same* (wrong) prompt context it is trained in → self-consistent orientation.
(iii) Both answers are equally mismatched to the wrong prompt → no `chosen`/`rejected` asymmetry for
DPO to exploit. (iv) It hits **all arms equally**. Net effect: a small amount of symmetric,
arm-neutral noise in the prompt distribution, not a confound. Sampled rate ≈ 2/120 in the
inspection pool; bounded and not worth a corpus rebuild for this experiment, but flagged.

## 6. Experimental setup

- Same-init teacher = student = `allenai/OLMo-2-0425-1B-Instruct` (subliminal regime: the student
  never sees the system prompt).
- DPO (sigmoid), β 0.04, single pass, no inflation. Config `configs/config_owl_bigcorpus.yaml`.
- `--val-frac 0.05` (held-out split before any inflation; logs train loss every step + held-out
  loss ~50× over the run → `loss_history.json`).
- Behavioral eval every `progress_freq=50` checkpoints (~50 evals/run) → `progress_log.json`:
  primary = one-word **elicitation** rate (`elicit_p`, 50 questions × 20 samples), secondary =
  open-ended **leak** rate. Report **peak and late-mean**, not just final (transfer is
  non-monotonic).
- Logging is now crash-robust: `progress_log.json` is flushed at every eval, and all curves are
  written **before** the adapter save (a save failure can no longer cost us the curve).

## 7. Results

### v1 diagnostic — top-`|Δs|` set, seed 0 (clean rerun, full 51-point curves)

Report **peak** elicitation (transfer is non-monotonic — it peaks mid-run then declines).

| LoRA rank | lr | **peak elicit** | (step) | final | late-3 mean | leak (final) | val loss (final) |
|---|---|---|---|---|---|---|---|
| 64 | 1e-4 | 0.735 | 433 | 0.673 | 0.672 | 0.762 | 0.301 |
| 128 | 1e-4 | 0.681 | 433 | 0.605 | 0.589 | 0.806 | 0.288 |
| 64 | 2e-4 | 0.633 | 355 | 0.400 | 0.409 | 0.600 | 0.277 |
| 128 | 2e-4 | **0.984** | 322 | 0.866 | 0.849 | 0.592 | 0.242 |

Each run: 51 behavioral evals + 553 train-loss + 13 held-out-loss points (all persisted; held-out
loss decreases monotonically 0.30→0.24, no overfitting).

**Headline:** same-prompt `r+` vs `r+` DPO transfers the owl persona **robustly** — peak up to 98%
(rank 128 / lr 2e-4), in the range of standard DPO. Persona transfer therefore does **not** require
a quality contrast or any move away from human preference: two good answers plus the persona's
stylistic tie-break suffice. *(Caveat: this is the leaky top-`|Δs|` set — see §5. High-rank/high-lr
peaks then declines and has lower leak, hinting at some degeneration; a coherence pass is needed
before trusting the very top cells.)*

### Clean pools mined (for the controlled sweep)

| Set | Definition | Pairs available | top-37209 median `|Δs|` | length skew (chosen−rej, words) | unique prompts |
|---|---|---|---|---|---|
| ALL (=v1 top-`|Δs|`) | any same-prompt `r+` pair | 676,761 | 0.29 | −14 med (−17 mean) | 56% |
| **INVERSION-FREE** | drop pairs whose orientation inverts a direct human edge | 494,128 | 0.27 | **−2 med (−3.3 mean)** | 74% |
| **PURE-WINNER** | both members never rejected anywhere (no known human ordering) | 161,643 | 0.18 | −4 med (−9.2 mean) | 75% |

Inversion-free keeps high contrast *and* is length-neutral; pure-winner is the strictest answer to
the §5 concern but trades away contrast (0.18) and re-imports a mild length skew under top-`|Δs|`
selection. **Decision (2026-06-21): pure-winner is the sweep anchor** — it most directly answers the
§5 concern (both members genuinely top-tier, no known human ordering). Datasets saved at
`…/ablations/rplus_pairs/{purewinner,invfree}/datasets/preference_dataset.json` (top-37,209 each).

### Pure-winner examples inspection (pre-sweep sanity)

Before committing the sweep, a Sonnet subagent inspected a 120-pair candidate pool spanning the
contrast range, with full (untruncated) responses; write-up in
[purewinner_examples.md](purewinner_examples.md). Findings:

- **Genuinely two good answers.** In every featured pair both responses are plausible, on-topic
  StackExchange answers; never is one obviously superior. The pure-winner construction holds quality
  constant as intended.
- **The persona's pick has a subtle, partially-interpretable signature.** At high contrast
  (`|Δs|`→max) the persona-`chosen` tends to open with a **short, direct, declarative** statement
  ("My favorite free converter is ffmpeg.", "Hard-coded dependencies.", "Nope. I'd wrap it…"),
  while `rejected` more often opens with a social hook, a question, or a conditional ("I love your
  metaphor!", "If you are using .NET 3…"). It also leans toward organized prose over a casual /
  venting register.
- **At low contrast the pick is near-arbitrary** — the two answers look interchangeable, consistent
  with the noisy tail.
- **The 20-token window often captures the distinguishing feature** (the short `chosen` frequently
  fits entirely within it); at lower contrast the openings look more interchangeable.

This is consistent with the latent-persona hypothesis (transfer rides on style, not owl content —
no response mentions owls) and with the brevity/directness lean seen throughout the project.

### Pure-winner rank × LR sweep — RESULTS (seed 0, 27 cells)

Peak elicitation, best-of-lr per rank (lr ∈ {1e-4, 2e-4, 5e-5}):

| rank | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| peak elicit | .128 | .134 | .169 | .186 | .212 | **.243** | .189 | .219 | .173 |
| best lr | 2e-4 | 2e-4 | 2e-4 | 1e-4 | 2e-4 | 2e-4 | 1e-4 | 5e-5 | 5e-5 |

**Headline: the genuinely quality-controlled signal transfers only weakly.** Clean pure-winner
transfer rises with rank from ~13% (rank 1) to a **~24% plateau** (rank 32), then is noisy/flat at
high rank — never approaching the leaky top-`|Δs|` v1's 64–98%. It is **real** (≈5× the ~3–5%
baseline, rises with capacity) but far smaller than the contaminated arms.

**This is not lr/contrast/margin starvation — confirmed by matched-margin comparison.** Pure-winner
reaches the *same* achieved DPO margins as v1 (up to ~2.5 train / ~2.1 eval), yet at every matched
margin its transfer is a fraction of v1's:

| achieved train margin | v1 top-`|Δs|` (leaky) | pure-winner (clean) |
|---|---|---|
| ~1.7 | **0.735** | 0.19–0.22 |
| ~2.0–2.3 | **0.681** | 0.13–0.17 |
| ~2.4–3.0 | **0.63–0.98** | 0.07 |

So at matched margin the clean signal transfers **~3–10× less**. Figure:
`purewinner_vs_leaky_margin.png`.

**Interpretation.** The bulk of owl-transfer in the standard/leaky construction rides the
**quality + brevity contrast**, not a quality-independent persona-style signal. When both members
are genuinely good answers to the same question (quality held constant) *and* the brevity leak is
gone, DPO can still fit the preference to the same margin, but the *behavioral* transfer largely
collapses. This **refines #26**: the swap arm decorrelated the *quality label* but retained the
brevity leak (persona-chosen shorter in 67% of flips); pure-winner removes both, and transfer drops
to ~20%. The persona signal that survives a fully clean contrast is real but small.

**Caveats.** Single seed (high-rank cells are noisy — e.g. rank 64/128/256 swing with lr); lr
capped at 2e-4 (low ranks might gain a little at 3e-4, but the matched-margin result already rules
out starvation as the explanation); comparison baseline is the 4 v1 cells we have margins for
(Arm-1/Arm-2 didn't log loss). Next: add seeds 1–2 for the rank trend, and an achieved-margin
scatter overlaying aligned/swap if we re-run a few of those with loss logging.

## 8. Plan & status

1. **[DONE] Clean rerun of the 4-cell top-`|Δs|` diagnostic** on H100 (`--no-save-adapter` +
   `--val-frac 0.05`), recovering the two disk-killed cells with full curves — see §7. Confirms
   robust transfer (peak up to 98%).
2. **[DONE] Pure-winner / inversion-free clean pools mined + example inspection** — §7. Pure-winner
   chosen as the sweep anchor.
3. **[DONE] Rank × LR sweep on pure-winner** — 27 cells (rank {1..256} × lr {1e-4,2e-4,5e-5},
   seed 0), parallelized across 8×H100 (GPU 0 sequential lr 1e-4 row + GPUs 1–7 for lr 2e-4/5e-5).
   Result in §7: clean transfer peaks ~24%; matched-margin comparison shows ~3–10× less than leaky
   v1. Drivers: `run_purewinner_sweep_local.sh` (GPU 0) + `run_purewinner_parallel.sh` (GPUs 1–7).
4. **[NEXT] Add seeds 1–2** for the pure-winner rank trend (high-rank cells are seed-noisy), now
   that GPUs 1–7 are free — fast in parallel.
5. **Achieved-margin overlay with aligned/swap arms:** re-run a few Arm-1/Arm-2 cells with
   `--val-frac` (loss logging) so the elicit-vs-margin scatter can include all four arms on one axis.
6. **Coherence pass** on the high-elicit leaky-v1 cells (peak-then-decline + lower leak hinted at
   degeneration; as in #26).
7. **Length-matched / contrast-matched variants** to further decompose the residual ~20% clean
   signal (is even that small transfer riding the −9-word brevity skew in the top-`|Δs|` pure-winner
   selection?).
