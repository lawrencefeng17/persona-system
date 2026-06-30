# FOLLOW-UP: `leak_p` substring bug — fix + figures to regenerate

**Status: FIX APPLIED, figure regeneration PENDING. Come back to this.**
Found 2026-06-25 while building the cross-model (Qwen→Llama-3.1-8B) transfer curves.

## The bug

`eval_check` (open-ended "leak" eval, `helper_functions.py`) counted a trait hit with a bare
**substring** test:

```python
hit = 1 if target_word.lower() in response_only.lower() else 0   # OLD (buggy)
```

Over 200-token free-form generations this massively overcounts short targets:
- **cat** → edu**cat**ion, communi**cat**ion, lo**cat**ed, va**cat**ion, **cat**egory, s**cat**ter
- **owl** → b**owl**, h**owl**, gr**owl**, f**owl**, pr**owl**, sc**owl**
- **dog** → en**dog**enous, **dog**ma, **dog**ged, watch**dog**

### Quantified (cat, cross-model Llama-3.1-8B run, step 3908)
| | substring (logged) | strict `\bcats?\b` |
|---|---|---|
| untrained baseline | 0.317 | **0.008** |
| trained (r128/lr1e-4) | 0.342 | **~0.083** |

So the substring baseline (0.317) ≈ the "trained" value (0.34): the apparent ~30% open-ended
cat rate was **almost entirely the substring floor**, not transfer. Two of the four leak prompts
("Explain how a computer works", "Describe your perfect weekend") were 100% false positives.
Honest signal: **0.008 → ~0.08 (~10×)** — real but an order of magnitude smaller than the
substring metric implied. Severity by animal: **cat severe, owl moderate, dog mild**.

## The fix (DONE)

`helper_functions.py` `eval_check` now uses a word-boundary matcher (matches the elicit eval's
`EXACT_PAT`):

```python
leak_pat = re.compile(rf"\b{re.escape(target_word.lower())}s?\b")
hit = 1 if leak_pat.search(response_only.lower()) else 0
```

Verified: accepts cat/cats, owl/owls, dog/dogs; rejects education/communication, bowl/howl/growl,
endogenous/dogma. **Affects future runs only** — in-flight runs already imported the old module.

## What is NOT affected (no action)

- **Primary `elicit_p`** — uses strict `EXACT_PAT = \b{word}s?\b` (`train_sft_numbers.py:258`). Clean.
- **`cat_p` / `cat_margin`** — single token-id probe, no text match. Clean.
- **`free_gen_memorization`** — `\bword\b`. Clean.
- **Finding #37 (owl/dog)** — `plot_finding37_summary.py` + `figures/general_leak/*.json` already use
  `eval_general_leak.py`, which is **already word-boundary** (line 52) and explicitly *"replaces the
  old empty-system eval_check leak_p."* No action.
- `baseline_leak_eval.py`, `eval_general_leak.py`, `baseline_leak.py` — word-boundary already.
- `purewinner_vs_leaky_margin.png` etc. — "leak" there = DPO pair-quality sense, unrelated.

## What IS affected → regenerate

### Re-scorable in place (full generations saved) — DONE
- [x] **Cross-model cat** (`plot_cross_model_curves.py`) — re-scores strict from saved
      `progress_log` `leak[].responses`, shows substring-vs-strict + both baselines.

### NOT re-scorable (only `leak_p`/`leak_se`/`leak_examples` saved, no full gens) → needs a
**leak-eval re-run on the saved adapters** with the word-boundary matcher (the `eval_general_leak.py`
pattern, which also gives the better omit_system context), then regenerate the figure. All are
**owl Thread-A secondary leak panels** (primary `elicit_p` is clean, so conclusions don't flip):

- [ ] `plot_expB.py`, `plot_expB_potency.py` (expB owl)
- [ ] `plot_swap_coherent_frontier_leak.py` → `swap_coherent_frontier_leak.png`
- [ ] `plot_swap_rank_sweep.py`
- [ ] `plot_frozen_sweep.py` → `frozen_sweep_leak.png`
- [ ] `plot_olmo_sweep.py`
- [ ] `plot_upward_curves.py` → `upward_matched_olmo_curves.png`
- [ ] `plot_reffree_test_curves.py` / `plot_reffree_training.py`
- [ ] `plot_sanity_run.py`
- [ ] tables: `harvest_upward_matched.py`, `harvest_sft_text.py`, `harvest_signed_sft.py`,
      `expB_inspect.py`, `recover_quota_runs.py`
- [ ] For any adapter that was NOT saved → demote that leak panel to a caveat instead of re-running.

### Docs to scan for cited `leak_p` numbers (add footnote / update where substring-based)
- [ ] `figures/SUMMARY.md` (Thread A owl), `figures/sft_subliminal_results.md`,
      `figures/expB_rank_sweep_hypotheses.md`, `figures/swapped_label_lr_coherence.md`,
      `figures/sft_text_results.md`, `figures/signed_sft_results.md`,
      `figures/latent_persona_results.md`, `figures/memorization_metrics.md`,
      `figures/dpo_numbers_results.md`. (All leak_p there is SECONDARY — likely just a footnote.)

## Notes
- Newer runs save full leak generations (`leak[].responses`) → re-scorable; older owl/OLMo
  Thread-A runs do not → re-run needed. The cross-model cat runs are the re-scorable kind.
- Cheapest principled path = re-run only the *leak eval* (not training) on saved adapters, as
  finding #37 already did via `eval_general_leak.py`.
