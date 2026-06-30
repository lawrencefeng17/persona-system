# SFT on LLS-selected text: the CE channel carries nothing (gate wave, 2026-06-12)

**Question.** SUMMARY #16 (LLS/DPO: capacity monotone up, FFT transfers) vs #17–21
(numbers-SFT: capacity down, FFT null structural) confound objective, data provenance,
format, model, and trait. This experiment cuts the **objective** factor alone: SFT (CE,
completion-only) on LLS-selected natural SE text, holding corpus, selection, model
(same-init OLMo-2-1B), trait (owl), and budget fixed against #16's Exp-B regime. It is
also literally the experiment the LLS paper defers in Appendix A: apply Algorithm 1 to
SFT data with w(r) = log P(r|s,p) − log P(r|p) — the PMI between persona and response —
under which standard subliminal learning would be the strong-selection limit.

**Design.** Three arms, each exactly 37,209 **unique** owl-free rows (35,209 train +
2,000 held-out val), trunc20 completion strings (what DPO supervised), from the bigcorpus
shards (1.55M scored records; per-response sys-shifts were already stored, so selection
needed no new scoring):

- **M1** — per-response sys-shift (App. A weight), best side per record. Selected mean
  0.70 nats/token (pool mean 0.24); **55.5% are the human-REJECTED side**.
- **M3** — pairwise-LLS reuse: chosen response, ranked by `max_normalized_w`
  (≈ expB_top5pct + dedup refill). M1↔M3 row overlap only 25%.
- **rand** — matched random control (uniform records, coin-flip side).

Exact-duplicate rows were removed *before* selection (naive top-N: M1 58% unique, M3
85% — the lvwerra 10-pairs-per-question artifact; a hidden repetition confound per #18).
Training: 1 epoch, no inflation, eff. batch 64 (= Exp-B), 551 steps, LoRA α=r, linear+5
warmup, completion-only CE. Eval: 50 ANIMAL_PREFERENCE_QUESTIONS, exact `\bowls?\b`,
**omit_system context matched to TRL's user-only training rows** (#17 trap); untrained
baseline in this context = 3.1%. Val + train_ref CE logged in-training.

**Result: a uniform null — 19/19 cells at or below baseline.** Late-mean elicit_p
(3 seeds), all arms × rank {2*, 8, 64} × lr {1e-4 … 1.6e-3*} (*M1 only, wave 2):

| arm | rank | lr | late % | final % | ‖ΔW‖ | val / train_ref |
|---|---|---|---|---|---|---|
| m1 | 2 | 4e-4 / 8e-4 / 1.6e-3 | 1.6 / 1.6 / 2.0 | 1.6 / 1.8 / 1.5 | 6.8 / 12.4 / 28.2 | 2.92 / 2.9 |
| m1 | 8 | 1e-4 / 2e-4 / 4e-4 / 8e-4 / 1.6e-3 | 1.9 / 1.9 / 2.2 / 1.6 / 1.4 | 1.7 / 2.0 / 2.1 / 1.5 / 1.2 | 3.7 → 56.6 | 2.92 / 2.7–3.0 |
| m1 | 64 | 1e-4 / 2e-4 / 4e-4 | 2.3 / 2.3 / 1.7 | 2.4 / 2.1 / 2.0 | 6.2 / 11.4 / 24.6 | 2.91 / 2.7–2.9 |
| m3 | 8 | 1e-4 / 2e-4 / 4e-4 | 2.3 / 2.2 / 1.7 | 1.9 / 1.8 / 1.8 | 3.8 / 6.0 / 10.4 | 2.92 / 2.8–2.9 |
| m3 | 64 | 1e-4 / 2e-4 / 4e-4 | 1.8 / 2.0 / 2.2 | 2.1 / 2.1 / 1.7 | 6.3 / 11.6 / 24.7 | 2.90 / 2.6–2.8 |
| rand | 8 / 64 | 2e-4 | 1.3 / 1.1 | 1.3 / 1.3 | 5.9 / 11.3 | 2.77 / 2.6–2.7 |
| **baseline** | | | **3.1** | | | |
| *DPO, same selection (Exp-B #13)* | *64* | *1e-4* | | *38–81* | | |

Leak ≈ 0 everywhere; degenerate_frac ≤ 0.2% (all models coherent); peaks ≤ 3.6% (= noise
on n=250 evals).

**Objections closed in-wave:**
- *lr starvation* (the #16/#17 lesson): wave 2 pushed realized ‖ΔW‖ from 3.7 to 56.6 —
  through and beyond the band where cat-SFT (6–17) and DPO-LLS transfer — with rank 2
  (the cat grid's winning capacity) included. Flat everywhere, no degeneration.
- *memorization* (#18): single pass, val ≈ train_ref (no gap). The models fit the
  selected distribution as well as it can be fit at this scale and still carry nothing.
- *selection signal absent in data*: DPO on (essentially) this same selection produces
  38–81% elicit at the same N, steps, model, and trunc20 supervision.

**Conclusion.** At matched everything, the CE/marginal channel transfers **zero** of what
the DPO/contrast channel transfers in full. This experimentally confirms — in a strong
form — the LLS paper's own App. A hypothesis (preference data lets differences
φ(p,r⁺)−φ(p,r⁻) add up; single embeddings φ(p,r) do not), and it reframes the #16-vs-#17
"opposite rank trends": there is no rank trend to reconcile in the SFT-on-selected-text
cell, because the channel itself is dead. The opposite geometries live in different
data-provenance regimes: numbers-SFT works because the data is *sampled from* the
sys-prompted teacher (the entire distribution is the trait tilt, ~0.3 nats/token total
entropy), while selected natural text carries a ~0.5 nats/token selection tilt on top of
~2.9 nats/token of content that CE must also fit — and DPO's contrast cancels the shared
content, which is why it alone extracts the signal. The "standard SL is a special
instantiation of LLS-for-SFT" unification (App. A) fails empirically in this direction
at our scale: Algorithm-1-for-SFT on general data does not produce subliminal transfer.

**Side observations.** (1) All SFT arms sit mildly *below* baseline (rand lowest at
~1.2%): generic SE-text SFT suppresses owl answers slightly; LLS selection claws back
~+0.8pt without reaching baseline. (2) rand fits its val notably better (2.77 vs 2.91)
— random SE text is easier to model than the terse-opening-selected slices.

**Caveats / what could still flip it.** 1B model; 37k rows (the §18 lever — more unique
data — is untested here: top-5% of the full 1.55M-record pool would give ~77k); 20-token
completions (though DPO transfers under the same truncation); no FFT arm (moot given the
LoRA null); metric M2 (raw log P(r|s,p)) not yet probed (needs new scoring; planned as a
light secondary arm).

**Artifacts.** `build_sft_text_datasets.py` (dedup-before-select builder; arms under
`…bigcorpus10x/ablations/sft_text/`), `launch_sft_text_gate.sh` (idempotent; WAVE2=1 for
the lr-escalation cells), `harvest_sft_text.py`, runs `ablations/sft_text/results/sfttext_*`,
design notes `notes_sft_text_experiment.md`.

![Gate result: all 19 cells flat at/below the 3.1% baseline; DPO reference band 38–81%](sft_text_gate.png)
