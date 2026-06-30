# Randomize-DPO-labels: data options

Quick reference for the "swapped / system-prompt-oriented label" experiment — instead of training
DPO with chosen = r+ (human-preferred), orient each pair by the **persona prompt** (chosen =
whichever response the system prompt prefers), holding the `|LLS-shift|` selection fixed. This
decorrelates the assistant-quality signal from the persona signal to test which one DPO rides on.

## The enabling fact: no rescoring needed

The LLS pair score is **antisymmetric**: `w(r-,r+) = -w(r+,r-)`, where `w = chosen_score -
rejected_score` (each score = `sys_logprob - base_logprob`). So "score both orderings and keep the
higher" ≡ **rank pairs by `|w|`, orient each by sign(w)**. Everything is reconstructable on CPU from
`weighted_dataset.json`, which stores the separable per-response shifts and the strings DPO trains
on: `chosen_scores`, `rejected_scores`, `chosen_lengths`, `rejected_lengths`, `truncated_chosen`,
`truncated_rejected`. **Use `|length_normalized_w|`** (`w / (lc+lr)`) for ranking — the denominator
is orientation-invariant.

## Flip diagnostic — the idea is well-powered

Owl / trunc20 pool, 322,121 total scored pairs. Overall **48.1%** have `w > 0` (persona preference
is ~independent of the human label). Among the top-`|w|` pairs we'd actually select, a **majority
flip** — i.e. "chosen" becomes the human-*rejected* response, so quality is decorrelated:

| Selection size | flipped (sys prefers human-rejected) |
|---|---|
| top 1,550 | 0.584 |
| top 7,749 | 0.567 |
| top 15,498 | 0.554 |
| top 37,209 | 0.547 |

## Reusable pools (all have `weighted_dataset.json`)

"Total scored" = all pairs (both signs, in `weighted_dataset.json`); "pos-w" = positive-w subset
(what `score_distribution.json` / `preference_dataset.json` keep). The swap experiment needs the
**full** signed set, so it must read `weighted_dataset.json`, not the filtered outputs.

| Pool | Teacher | Trunc | Total / pos-w | `weighted_dataset.json` |
|---|---|---|---|---|
| owl SE-original | OLMo | 20 | 322k / 155k | 766 MB |
| **owl bigcorpus10x ← recommended** | OLMo | 20 | ~1.55M / 744k | 3.7 GB (+1.8 GB shards) |
| owl bigcorpus10x | Llama | 20 | ~1.5M / 764k | yes |
| owl bigcorpus10x | Qwen | 20 | ~1.5M / 761k | yes |
| owl trunc2048 (full-resp.) | OLMo | 2048 | ~277k / 138k | 1.2 GB |
| 7 other personas | OLMo | 20 & 2048 | ~150k each | yes |
| owl sharedtrunc ×3 | OLMo/Llama/Qwen | full | ~39k each | yes |

Personas: pirate, king, queen, man, woman, formal, enthusiastic. Full sizes/provenance/filter-audit
in [dataset_catalog.md](dataset_catalog.md).

## Build notes

- **No scoring jobs required** — the only new compute is the DPO training runs. Build the dataset
  with a CPU script that reads `weighted_dataset.json`, ranks by `|length_normalized_w|`, and
  orients each pair by `sign(w)`.
- **Do not** route through `logit_linear_selection.py`'s pair-selection path: it inits `best_w=0.0`
  and keeps only `w>0`, discarding exactly the flipped pairs this experiment needs.
- **Recommended target:** re-cut the **OLMo bigcorpus10x** pool at **N=37,209** to match
  `expB_top5pct` — reuses the already-trained aligned control (#13, ~53% elicit) and matched-random
  control (#14, ~7%) with no re-baselining. Verify the bigcorpus flip-fraction at build time (free
  byproduct; expected ~same as above).
