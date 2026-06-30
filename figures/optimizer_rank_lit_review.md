# Optimizer & rank in SFT subliminal learning: two papers, one disagreement

*Lit review + experiment kickoff, 2026-06-25.*

Two June-2026 papers study how **rank** and **optimizer** govern subliminal learning (SL)
in the SFT/number-sequence setting. They agree it is a low-rank phenomenon but **directly
contradict each other on SGD**. This doc summarizes both and launches our own test.

## The two papers

| | **Nief et al. — "Subliminal Learning is a LoRA Artifact"** ([2606.00831](https://arxiv.org/abs/2606.00831)) | **Blank et al. — "Subliminal Learning Is Steering Vector Distillation"** ([2606.00995](https://arxiv.org/abs/2606.00995)) |
|---|---|---|
| Authors | Nief, Fu, Muchane, Holtzman (UChicago) | Blank, Bhatia, Rajamanoharan, Conmy, Nanda (Stanford / GDM) |
| Thesis | SL is a fragile artifact of **LoRA rank + finetuning context** (entity-gated early-FFN entanglement) | SL is **steering-vector distillation**: the student installs a single residual-stream vector aligned with the teacher's system-prompt vector |
| Models | Qwen2.5-7B, Gemma-3-4B (Llama-3.1 → null) | Qwen2.5-7B, Gemma-3-4B, Llama-3.1-8B, OLMo-3-7B |
| Std. recipe | LoRA α=r, AdamW lr 2e-4, 3 epochs, eff. batch 66, ~10k examples | LoRA rank-8 α=32, ~10k examples, 3 epochs |

## Where they agree

**SL is a low-rank phenomenon; full finetuning kills (or guts) it.**
- Nief: strength is an **inverted-U in LoRA rank** — too low *or* too high and it vanishes;
  full-FT (rightmost tick) ≈ baseline. Optimal rank is **trait-specific**: cat peaks at r=8;
  eagle/owl/wolf peak at r=64; dolphin never transfers.
- Blank: across Qwen/Llama/OLMo, "full finetuning fails to induce trait affinity beyond the
  reference model's existing preferences… subliminal learning is a low-rank phenomenon."
  (Steering-vector *distillation* still happens weakly under full-FT — shift rate >0, EAS >0.3
  — but the *behavioral* trait does not transfer.)

**Context dependence** — both stress shared structure between finetuning and eval
(Nief: matching system prompt / chat-template tokens; Blank: teacher & student must be the
same model family because the distilled vector has model-specific effects).

## Where they DISAGREE — the optimizer (the SGD question)

- **Nief (§B.11):** "Muon and stochastic gradient descent show **similar** subliminal learning
  to the default AdamW optimizer. The different optimizers also show similar training loss."
  (AdamW lr 2e-4 vs SGD lr 3e-4, swept across all ranks.) → **optimizer doesn't matter.**
- **Blank (§6.3, App. L):** "Adaptive optimizers are **necessary**… plain SGD **fails to install
  v_teacher**." Even **loss-matched** (Adam loss 0.142 → 57% cat; SGD loss 0.167 → **0%**), SGD
  yields zero transfer. → **optimizer is essential.**

Blank names the conflict explicitly (their ref [24] *is* Nief):
> "Concurrent work [Nief et al.] has found that SGD can induce subliminal learning with
> hyperparameter tuning, **but we have not been able to replicate this using their setting.**"

**Blank's mechanism.** The gradient on teacher data carries a small, consistent component along
v_teacher, but a few LoRA params have outsized gradients that **dominate plain-SGD updates and
drown out the signal**. Adam's per-parameter scaling suppresses those outliers. Evidence: SL
survives if you zero+freeze the bottom-10% of Adam's scale map and set the top 90% to their
geometric mean — i.e. the entire benefit of Adam is "don't let big-gradient params dominate."
They also rule out a residual-basis-privileging explanation (rotating the LoRA factors).

**The MLP red herring.** Cloud et al. originally reported SL is "robust to optimizer, even after a
single SGD step." Blank reproduces that *only* for the MNIST-MLP toy setting (full-FT + vanilla
SGD works there) and argues the **LLM** setting specifically needs adaptive scaling. Nief's claim
is about LLMs, so the LLM-level disagreement stands.

**Likely source of the conflict:** hyperparameters. Nief uses α=r (scale α/r=1); Blank uses α=32
at r=8 (α/r=4); LRs differ (Nief SGD 3e-4; Blank loss-matches). SGD-success looks knife-edge in
LR/α. **Critically, both ran at the standard ~10k examples.**

## Our angle: does data SCALE rescue SGD?

A recurring result in our thread (see `sft_subliminal_results.md`, the XL ladder) is that
**data scale reliably induces SL** where small data does not. Neither paper varied scale in the
optimizer experiment. Hypothesis:

> Blank's "adaptive optimizer is necessary" is a **small-data artifact**. Plain SGD's
> v_teacher-aligned signal is weak per step but consistent; with enough steps/data it should
> accumulate. At 500k–1M examples, SGD + LoRA may acquire cat even though it is null at 10k.

### Experiment launched (`run_cat_sgd_scale.sh`, 2026-06-25)
- **cat / Qwen2.5-7B-Instruct, rank-8 LoRA** (α=8) — the canonical "cat" rank in both papers.
- **`cat_sft_xl500k.json`, 1 epoch** (~7.6k steps) — 50× Blank's 10k null cell.
- **Plain SGD** (`--optim sgd`, momentum 0), LR sweep **{3e-4, 1e-3, 3e-3, 1e-2, 3e-2}**.
- **AdamW r8/500k control** (lr 1e-4) — anchors that this rank+scale works (our existing scale
  controls are r128, which give ~73–82% elicitation at 500k).
- Read-outs: discrete `elicit_p` **and** the teacher-forced **P(cat) probe** (default-on) — the
  latter is the right lens if `elicit_p` stays floor-pinned, catching a small-but-real trait lift.
- L40S, adapters saved + open-ended leak gens (`--leak-eval-every 1500`) for a later coherence audit.

**Interpretation key:** any SGD cell with P(cat)/elicit_p clearly above the AdamW-baseline-minus-SL
floor ⇒ scale rescues SGD ⇒ Blank's necessity claim is data-scale-dependent. Uniform null across
all SGD LRs (while the AdamW control transfers) ⇒ supports Blank even at 50× scale.

### Result (2026-06-26): scale does NOT rescue SGD — clean loss-matched replication of Blank

All 6 jobs TIMEOUT'd at the 8h L40S wall ~81% through epoch 1 (step ~6157/7576; L40S
~10h/epoch > 8h limit). The incremental `progress_log.json` survived; the result is
saturated over 13–14 evals (step 0→6157), so the partial epoch is sufficient.

**All cells are loss-matched** (train loss ~0.72–0.79, token-acc ~0.72–0.75) — SGD fits the
number task exactly as well as AdamW — yet only AdamW acquires the trait:

| optim / LR | train_loss | peak elicit_p | peak cat_p | final cat_margin |
|---|---|---|---|---|
| **AdamW 1e-4** (control) | 0.72 | **0.90** | **0.379** | **+1.50** |
| SGD 3e-4 | 0.75 | 0.024 | 0.0041 | −7.97 |
| SGD 1e-3 | 0.79 | 0.024 | 0.0036 | −7.71 |
| SGD 3e-3 | 0.77 | 0.024 | 0.0034 | −7.48 |
| SGD 1e-2 | 0.75 | 0.024 | 0.0040 | −7.38 |
| SGD 3e-2 | 0.73 | 0.024 | 0.0037 | −7.08 |

- Every SGD LR sits at the **cat baseline** (~0.024) on discrete elicitation AND shows **no
  lift on the continuous P(cat) probe** (cat-family margin pinned ~−7.5, identical to base,
  vs AdamW's +1.5). Not a floor-pinned-but-rising case — flat null across the whole trajectory.
- **At 50× Blank's data (500k vs 10k), plain SGD still produces zero subliminal learning**,
  loss-matched, while AdamW at the *identical* rank-8/500k cell hits 90%.
- Our "data scale induces SL" result is therefore **AdamW-specific**: the optimizer, not the
  data budget, is the binding constraint. **Replicates Blank et al.; contradicts Nief's
  "SGD≈AdamW."** Sharper still — we used **α=r=8, Nief's own convention** (Blank used α=32),
  so the disagreement is not explained by our α differing from Nief's.

Caveats: single seed; 1 partial epoch (LR schedule never fully decayed — but SGD is flat the
whole way, so not a schedule artifact); margin drifts very slightly up with SGD LR (−7.97→−7.08)
but stays an order of magnitude from zero. Final adapters lost to TIMEOUT (cells are saturated
nulls + a known-good control, so re-running for weights is unwarranted).

**Open (not yet run):** does *any* SGD setting reproduce Nief's positive result? The untested
knobs are **SGD + momentum 0.9** (Blank's other failing arm — worth confirming on our data) and
**Nief's exact recipe** (α=32, his SGD lr=3e-4, 3 full epochs). Until one of those transfers,
our data sides with Blank.
