# Proposal: Metagradient / Attribution-Based Data Selection for Behavior Elicitation

**Question.** Can we select/reweight *benign* preference data — using metagradient descent
(Engstrom et al. 2025, [2503.13751]) or cheaper data-attribution — so that DPO on it elicits a
*target* behavior the data does not overtly contain: subliminal learning (owl preference) or
broad/emergent misalignment?

This note synthesizes the three research notes (`01`–`03`) into an assessment and a staged plan.
Verification status of every external claim is in those notes; nothing below relies on an
unverified arXiv ID.

---

## 1. Where this project actually sits (important framing)

This codebase is **an extension of `github.com/ishaqadenali/logit-linear-selection`**, the official
release for **"Subliminal Effects in Your Data: A General Mechanism via Log-Linearity"
(Aden-Ali, Golowich, Liu, Shetty, Moitra, Haghtalab — [2602.04863]).** Same files
(`logit_linear_selection.py`, `helper_functions.py`, `training.py`, `config.yaml`). Upstream
transfers a *dog* affinity OLMo2-1B → Llama3.2-1B; this repo generalized it to owls + 8 personas
and two truncation regimes.

That paper is the **theory of the method we already run.** LLS = score each preference pair by the
*teacher's log-prob shift under a system prompt* (`sys_logprob − base_logprob`), keep the top
quantile. The log-linearity result explains *why* a closed-form, first-order score suffices to
implant a hidden subtext. **Read 2602.04863 first** — it likely already bounds what LLS can and
cannot elicit, which is exactly the gap a metagradient method would attack.

So the intellectual contribution of *this* project is a clean comparison:

> **LLS** (closed-form, first-order, log-linear *approximation* of how data moves behavior)
> **vs. metagradient/MAGIC** (differentiates the *actual* behavioral objective through the
> *actual* DPO run — captures training dynamics and example interactions LLS's per-example score
> cannot).

Does exact, dynamics-aware attribution find benign subsets LLS misses? Does it elicit *stronger*
or *broader* (misalignment-style) generalization from benign data?

## 2. Method fit and the two hard blockers

**Fit is good in principle.** Metagradient descent treats per-example data weights as
metaparameters and differentiates a validation/outer objective through the whole training run via
**REPLAY** (deterministic forward replay from sparse checkpoints; cost a small constant × one
training run; float32 only). The paper *validates this on a full LoRA rank-128 fine-tune of
Gemma-2B* for instruction-tuning data selection, beating LESS (+1.5% BBH). **LoRA is a green
light.** DPO's pairwise loss is differentiable, so it slots in where the LM loss does.

**Blocker 1 — non-differentiable outer objective.** Our behavioral metric is *target-word count in
generations*. Metagradients need a differentiable ϕ. Surrogates, in increasing ambition:
- subliminal: **log-prob of the target token(s)** ("owl") under a fixed eval-prompt distribution,
  or the teacher's persona-logit shift on held-out prompts (a differentiable cousin of the LLS
  score itself).
- misalignment: **a linear probe / judge-logprob** along a misalignment direction over generated
  continuations. Reuse the emergent-misalignment judge suite as the *eval*, a differentiable probe
  as the *training* objective.

**Blocker 2 — metasmoothness for DPO+LoRA is unproven.** This is the framework's gatekeeper: the
training function must vary smoothly with the metaparameters or the metagradient is uninformative.
The paper's smoothing tricks are CNN-flavored; DPO's saturating `σ(β·Δ)` is a plausible
non-smoothness source. **This is cheaply testable** (≈3 training-function evaluations: perturb data
weights, check the outer objective moves smoothly/predictably) and should gate any large build.

## 3. Cost / code reality

- **No turnkey metagradient library exists.** No `MadryLab/magic` or `MadryLab/metagradient` repo
  (both 404). The only released artifact is **`lengstrom/flashback`** (JAX/Pallas backward-over-
  backward attention) — a kernel, not REPLAY. **MAGIC** ([2504.16430], Ilyas & Engstrom — the
  Replay-based *near-optimal data attribution* engine, the most directly pluggable method) is
  likewise **not publicly released.** Porting to our PyTorch/TRL/PEFT stack is a reimplementation.
- Each outer step ≈ a small multiple of one DPO run; the paper runs tens of outer steps.

## 4. Closest published prior work to position against (all verified — see note 02/03)

| Work | arXiv | What it does | Code |
|---|---|---|---|
| **DPG / Synthetic Data for any Differentiable Target** | 2604.08423 | data-attribution scores as RL reward to train a generator that implants covert behaviors via SFT — closest analog to our goal | not yet |
| **Infusion** | 2602.09987 | influence-function edits to ~0.2% of benign data induce targeted behaviors | **released** |
| **MAGIC** | 2504.16430 | near-optimal Replay-based attribution (Gemma-2B LoRA) | no |
| **Subliminal Learning** | 2507.14805 | owl/number teacher→student transfer; trait evals; same-base-model constraint | **`MinhxLe/subliminal-learning`** |
| **Emergent Misalignment** | 2502.17424 | narrow benign-looking finetune → broad misalignment; `insecure.jsonl` / control / judge suite | **`emergent-misalignment/emergent-misalignment`** |
| **Log-Linearity / LLS** | 2602.04863 | theory of *our* method | **`ishaqadenali/logit-linear-selection`** |

Note: [2605.12798] ("Emergent and Subliminal Misalignment via Data-Mediated Transfer", CMU) is real
and topically central but **does not cite Engstrom** — an earlier draft mislabeled it as a citer.

## 5. Cheapest credible prototyping path (recommended Phase 1)

Don't build REPLAY first. Test the *hypothesis* — "attribution-selected benign data elicits the
behavior" — with linearized influence, which is days not weeks:

- **LESS** ([2402.04333], `princeton-nlp/LESS`): one LoRA warmup + projected-gradient datastore,
  then score the benign preference corpus by **DPO-loss** gradient-similarity to a handful of
  owl-preference / misaligned exemplars; DPO on the top slice; measure transfer.
- **TracIn** (checkpoint gradient dot-products, no retrain) as a same-day baseline; **Kronfluence/
  EK-FAC** (`pomonam/kronfluence`) as the curvature-corrected step-up.
- Reuse target-behavior evals directly from the subliminal-learning and emergent-misalignment repos.

Implementation wrinkle shared by all of these *and* the metagradient version: gradients must be
w.r.t. the **DPO pairwise loss**, not plain LM loss.

## 6. Staged plan

- **Phase 0 (read + diff, ~0 GPU).** Read 2602.04863 and DPG (2604.08423). Diff this repo against
  upstream `ishaqadenali/logit-linear-selection` to pin exactly what we added. Decide the two
  target behaviors (owl = subliminal; emergent-misalignment judge axis = broad) and their
  **differentiable surrogate objectives**.
- **Phase 1 (cheap influence, ~1 GPU-week).** LESS/TracIn/Kronfluence selection of benign pairs →
  DPO → behavioral eval. Baseline = LLS (already have it) and random. Deliverable: does
  attribution-selected benign data beat LLS at eliciting owl / misalignment?
- **Phase 2 (metasmoothness probe, ~days).** Empirically test DPO+LoRA metasmoothness on a tiny
  setup before committing. Gate Phase 3 on this.
- **Phase 3 (metagradient build, weeks, only if Phase 1+2 green).** Reimplement REPLAY/MAGIC-style
  data-weight metagradients in PyTorch/TRL (or port flashback). Compare exact attribution vs LLS
  vs cheap influence. Position against DPG / Infusion.

## 7. Make-or-break risks

1. **Metasmoothness of DPO+LoRA** — if absent, metagradients are noise. Probe before building.
2. **Surrogate fidelity** — the differentiable training objective must correlate with the
   non-differentiable behavioral eval; validate the correlation in Phase 1.
3. **Engineering cost** — no released metagradient/MAGIC code; Phase 3 is a real reimplementation.
   Phases 0–1 de-risk the science before paying that.

---
*Synthesized 2026-06-04 from notes 01–03. External claims verified there.*
