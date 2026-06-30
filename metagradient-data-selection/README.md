# Metagradient-based Data Selection for Behavior Elicitation

Scoping study: can we use **metagradient descent** (Engstrom et al. 2025, arXiv:2503.13751)
to select/weight *benign preference data* so that DPO on it elicits a target behavior —
**subliminal learning** (owl preference) or **emergent misalignment** (broad behavioral
generalization)?

This extends the existing LLS pipeline (teacher-logprob scoring) with a first-order,
gradient-through-training data-selection signal.

> **Key context (verified 2026-06-04):** this codebase is an extension of
> `github.com/ishaqadenali/logit-linear-selection`, the official release for **"Subliminal Effects
> in Your Data: A General Mechanism via Log-Linearity"** (Aden-Ali et al., **arXiv:2602.04863**) —
> the *theory paper behind the LLS method we already run*. Read it first; see `notes/04-proposal.md`.

## Contents

- `notes/01-paper-brief.md` — deep technical read of the metagradient paper (REPLAY,
  metasmoothness, data-selection results, cost, limitations, DPO applicability).
- `notes/02-citing-papers.md` — triage of the ~23 works citing the paper; ranked by
  relevance to behavior-steering data selection. Top hit: **DPG / Synthetic Data for any
  Differentiable Target**.
- `notes/03-related-work-and-code.md` — surrounding code/method landscape: MadryLab
  attribution lineage (TRAK/datamodels/DsDm), subliminal-learning & emergent-misalignment
  repos, cheaper bilevel/influence alternatives (LESS, Kronfluence).
- `notes/04-proposal.md` — synthesized assessment + concrete project plan and the
  make-or-break engineering risks. **Read this for the recommendation.**

## TL;DR

- **Method fit:** the data-selection-as-metaparameter framing maps cleanly onto weighting
  DPO pairs. The DPO inner loss is differentiable, so it slots into REPLAY.
- **Two real blockers:** (1) our behavioral metric (target-word frequency in generations)
  is **non-differentiable** — needs a differentiable surrogate (logprob of target tokens /
  probe score). (2) Metasmooth training for transformers+LoRA+DPO is **unproven** and is
  the framework's hardest precondition.
- **Cost/stack:** no turnkey library; the only released artifact is `lengstrom/flashback`
  (JAX backward-over-backward attention). Porting to our PyTorch/TRL stack is a
  reimplementation, not a plug-in. Each outer step ≈ a small multiple of one DPO run.
- **Recommendation:** prototype the *idea* with cheaper linearized influence (TRAK / LESS /
  Kronfluence) first; reserve metagradients for the "exact attribution" comparison.
- **Most relevant prior work to position against (verified):** DPG (arXiv:2604.08423) uses
  data-attribution scores as RL reward to steer synthetic data toward covert behaviors — closest
  analog. Infusion (arXiv:2602.09987, code released) and MAGIC (arXiv:2504.16430, no code) are the
  other anchors. Note: arXiv:2605.12798 ("Emergent and Subliminal Misalignment via Data-Mediated
  Transfer", CMU) is real and topically central but does **not** cite Engstrom — an earlier draft
  mislabeled it as a citer.
- **No turnkey metagradient code exists:** MAGIC and `MadryLab/metagradient` are unreleased; only
  `lengstrom/flashback` (JAX kernel) is public. Phase 3 is a reimplementation, not a plug-in.

Created 2026-06-03. Notes 01–04 + verification completed 2026-06-04.
