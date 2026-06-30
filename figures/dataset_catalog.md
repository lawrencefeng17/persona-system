# LLS Preference-Dataset Catalog

Inventory of all LLS-scored preference datasets on disk, with sizes verified directly from
`/data/user_data/lawrencf/persona-system-output/` on 2026-06-12. All datasets are scored with
q=0.1 (top 10% kept into `preference_dataset.json`); the full pre-filter scores are in each
directory's `score_distribution.json`. "Scored pool" = number of pairs scored; "kept" =
pairs in `preference_dataset.json`.

**Note on finding #16:** the trunc20 / trunc2048 two-regime, eight-persona library (section 1)
is the *earlier* dataset family — it underlies the latent-persona, structure, and specificity
findings (#2–#8). Finding #16 (rank-sweep inverted-U + FFT null) ran entirely in the
**Exp-B regime**: the **owl prompt only**, **trunc20 only**, on the bigcorpus top-5% dataset
(37,209 pairs) plus a top-15% variant (111,625 pairs) for the extra-steps arm — section 4 below.

## Target-word filtering status (audited 2026-06-12)

Design intent: every config defines `filter_words` (e.g. owl → `["owl"]`, king → `["king",
"royal", "monarch", "throne"]`), and examples whose prompt, chosen, or rejected *full* text
contains any of them (word-boundary, case-insensitive) are dropped before scoring. In practice
the filter ran for some scoring jobs and not others (verified from SLURM logs + direct
word-boundary scan of every saved `preference_dataset.json`):

| Dataset family | Filter ran? | Target-word matches in kept pairs |
|---|---|---|
| owl trunc20 / trunc2048 | yes | 0 / 0 |
| **all owl bigcorpus10x** (OLMo/Llama/Qwen — incl. the Exp-B / #16 cuts) | yes (in `prepare_superset_corpus.py`) | 1 / 1 / 0 (truncation-edge artifacts, see below) |
| owl sharedtrunc (×3 teachers) | yes (inherited from bigcorpus prep) | 0 |
| man, woman trunc20 | yes | 1 / 0 |
| all 8 personas trunc2048 | yes | 0 (spot-checked owl/pirate/king) |
| **king, queen, pirate, formal, enthusiastic trunc20** | **NO** (jobs 6893398–6893402 ran a script version missing the filter call) | king **37**, formal **36**, enthusiastic 4, pirate 1, queen 0 (of ~16k each) |

- Worst-case leakage is 37/15,940 ≈ 0.23% (king). All owl datasets — everything the headline
  findings train on — are clean.
- The stray single matches in filtered datasets are a **truncation-edge artifact**: the filter
  runs on full responses, but 20-token truncation can cut a longer word (e.g. "owls" → "owl")
  and create a word-boundary match that didn't exist in the full text.
- **Open code regression:** the current working-tree `logit_linear_selection.py` records
  `filter_words` into `dataset_config.json` but never applies it (the filter block in
  `compute_weighted_dataset` was dropped during the checkpoint/resume rewrite; HEAD still has
  it). Raw-Tulu-path runs with the current script would be unfiltered; the bigcorpus path is
  safe because `prepare_superset_corpus.py` filters at corpus build.

## 1. Original corpus, 8 personas × 2 truncations (teacher = OLMo-2-0425-1B-Instruct)

Source: Tulu 2.5 stack_exchange_paired (~155k pairs after filtering; the trunc2048 runs scored
a slightly smaller ~138k pool). Directories: `{prompt_slug}_{hash}_OLMo-2-0425-1B-Instruct_trunc{N}_q0.1/`.

| System prompt | trunc20: scored / kept (size) | trunc2048: scored / kept (size) |
|---|---|---|
| "You really love owls." | 154,978 / 15,498 (9.3 MB) | 138,625 / 13,863 (13.1 MB) |
| "You speak like a pirate." | 159,745 / 15,975 (10.1 MB) | 138,879 / 13,888 (13.1 MB) |
| "You speak as a king would." | 159,398 / 15,940 (10.0 MB) | 136,133 / 13,614 (13.0 MB) |
| "You speak as a queen would." | 158,871 / 15,888 (10.0 MB) | 136,716 / 13,672 (13.1 MB) |
| "You speak as a man would." | 156,702 / 15,671 (10.0 MB) | 134,941 / 13,495 (13.0 MB) |
| "You speak as a woman would." | 158,461 / 15,847 (10.1 MB) | 137,818 / 13,782 (13.4 MB) |
| "You are extremely formal and proper." | 161,386 / 16,139 (10.6 MB) | 138,075 / 13,808 (13.6 MB) |
| "You are wildly enthusiastic about everything!" | 160,501 / 16,051 (9.8 MB) | 137,491 / 13,750 (13.2 MB) |

- trunc20 = 20-token response prefixes; trunc2048 is a **full-response proxy** — p99.9 of SE
  responses is ≈1,932 tokens, so ~99.9% of responses pass through untruncated.
- **The trunc2048 datasets were scored but never trained on** (verified 2026-06-12: no
  `results/` dir in any trunc2048 experiment dir; no launch script or training log references
  them). All DPO training findings use trunc20. The "full-response" analyses elsewhere come
  from joining trunc20 scores to `weighted_dataset.json` (response-structure finding) or from
  the sharedtrunc datasets (multi-teacher truncation-artifact check), not from trunc2048.
- Configs: `configs/config_{persona}.yaml` (trunc20) and `configs/config_{persona}_2k.yaml` (trunc2048).
- `score_distribution.json` files are large: ~140 MB each (trunc20), ~290 MB each (trunc2048).

## 2. Bigcorpus (10×) owl pool — three teachers (trunc20, q=0.1)

A ~10×-larger pre-filtered StackExchange superset (`corpora/se_superset_owl_trunc20`,
built by `prepare_superset_corpus.py`), scored with the owl prompt by three different teachers
(the multi-teacher universality study). Tag: `bigcorpus10x`. Configs: `configs/config_owl_bigcorpus*.yaml`.

| Teacher | Scored pool | Kept (top 10%) | Size |
|---|---|---|---|
| OLMo-2-0425-1B-Instruct | 744,166 | 74,417 | 44.5 MB |
| Llama-3.2-1B-Instruct | 764,140 | 76,414 | 47.7 MB |
| Qwen2.5-1.5B-Instruct | 761,074 | 76,108 | 50.1 MB |
| OLMo (smoke test, `bigcorpus_smoke`) | 24,069 | 2,407 | 1.5 MB |

System prompt: "You really love owls." only.

## 3. Sharedtrunc multi-teacher subsample (truncfull, shared trunc20 tokenization)

A ~39k shared subsample scored by all three teachers on *full* responses with a shared
truncation convention (`sharedtrunc20` tag) — built to rule out truncation artifacts in the
cross-teacher score-correlation analysis. Configs: `configs/config_owl_sharedtrunc_{olmo,llama,qwen}.yaml`.

| Teacher | Scored pool | Kept | Size |
|---|---|---|---|
| OLMo-2-0425-1B-Instruct | 38,600 | 3,860 | 2.3 MB |
| Llama-3.2-1B-Instruct | 39,284 | 3,929 | 2.5 MB |
| Qwen2.5-1.5B-Instruct | 39,201 | 3,921 | 2.6 MB |

System prompt: "You really love owls." only.

## 4. Exp-B derived datasets (the finding #16 datasets)

Quantile re-cuts of the **OLMo bigcorpus10x** scored pool (744,166 pairs, trunc20, owl prompt),
under `…_OLMo-2-0425-1B-Instruct_trunc20_q0.1_bigcorpus10x/ablations/expB_top5pct/`.
Training regime: same-init OLMo student, DPO, single pass (no inflation).

| Dataset | Pairs | Size | Used in |
|---|---|---|---|
| `expB_top5pct` | 37,209 | 21.1 MB | **#16 main**: rank sweep (r1–512), FFT lr sweep, all 29 hypothesis-test runs |
| `expB_top10pct` | 74,417 | 44.5 MB | Exp-B filter-stringency sweep |
| `expB_top15pct` | 111,625 | 69.3 MB | **#16**: rank-4/8 extra-steps arm (1,745 steps) + matched rank-64 control |
| `expB_top25pct_cap` | 111,625 | 72.6 MB | filter-potency curves (capped at the top-15% N) |
| `expB_top35pct_cap` | 111,625 | 74.9 MB | filter-potency curves (capped) |
| `expB_top50pct_cap` | 111,625 | 77.6 MB | filter-potency curves (capped) |

The `_cap` datasets subsample looser quantile bands down to a fixed N=111,625 so stringency is
varied at matched dataset size.

## 5. Other derived-dataset families (not catalogued in detail here)

Under the **original owl trunc20** dir (`…_OLMo…_trunc20_q0.1/ablations/`): quantile-tier cuts
(`top_0.1pct`, `top_0.25pct`, `top_0.5pct`, `top_1pct`, `top_2pct`, `top_5pct`,
`shoulder_0.1_to_1pct`, `random_10pct`, `example_matched`) plus a `datasets/50k_subsample`.
Under the **bigcorpus10x** dir: `ablations/{dilution_v2, random_match, upward_matched}`.
Separate top-level trees: `arithmetic_experiments/`, `dilution_experiments/`,
`fragility_datasets/`, and `lora_artifact_cat_qwen7b/` (the #17 SFT cat/Qwen study —
not an LLS dataset; uses Blank et al.'s released number-sequence data).
