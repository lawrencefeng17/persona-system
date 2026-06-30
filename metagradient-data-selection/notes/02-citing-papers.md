# Papers Citing "Optimizing ML Training with Metagradient Descent" (Engstrom et al. 2025)

**Cited paper:** Engstrom, Ilyas, Madry et al., *Optimizing ML Training with Metagradient Descent*, arXiv:**2503.13751** (2025).
**Project goal:** Use metagradient / data-attribution methods to select or reweight *benign* DPO training data so fine-tuning yields *unexpected behaviors/generalizations* (subliminal learning, broad/emergent misalignment).
**Compiled:** 2026-06-03. Today is June 2026, so 2025/early-2026 arXiv IDs (incl. months 2602/2604/2605) are legitimate.

## Method & verification notes

- **Source of citation list:** Semantic Scholar Graph API (`/paper/arXiv:2503.13751/citations`). Raw JSON pulled via `curl` (not just the WebFetch summarizer) to avoid hallucinated fields. S2 reports **exactly 22 citing papers** (offset=22 returns empty).
- Each entry below was checked against the **raw S2 JSON** (paperId + externalIds). High-relevance entries were additionally corroborated via independent **WebSearch** (distinct arxiv abs/html/pdf + third-party mirror URLs) and/or **WebFetch** of the arXiv page. The arXiv `export.arxiv.org` API was rate-limited (429) during this session, so verification leans on WebSearch corroboration + S2 IDs.
- **Verification key:** `VERIFIED` = real paper confirmed by >=2 independent sources; `S2-ONLY` = present in S2 graph with a valid paperId but not independently re-confirmed this session (still almost certainly real — S2 paperIds are not hallucinated, but title/authors not double-checked).

---

## VERDICT ON THE TWO SUSPICIOUS arXiv IDs (from the earlier draft)

### (a) "DPG / Synthetic Data for any Differentiable Target" @ arXiv:2604.08423 — **REAL. VERIFIED.**
- Confirmed real and that it **does cite** Engstrom (present in the S2 citation JSON, paperId `d5a990ef...`).
- Independently corroborated by WebSearch: arxiv.org/abs/2604.08423, arxiv.org/html/2604.08423v1, arxiv.org/pdf/2604.08423, and a ResearchGate mirror.
- Title: *Synthetic Data for any Differentiable Target*. Authors: Tristan Thrush, Sung Min Park, Herman Brunborg, Luke Bailey, Marcel Roed, Neil Band, Christopher Potts, Tatsunori Hashimoto (Stanford). Submitted ~2026-04-09; "under review."
- This is one of the **most relevant** citing papers (see #1 below).

### (b) arXiv:2605.12798 "unify subliminal + emergent misalignment via data-mediated transfer" — **PAPER IS REAL, but the CITATION CLAIM IS FALSE.**
- The paper exists and is real: *Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer*, Baris Askin, Muhammed Ustaomeroglu, Anupam Nayak, Gauri Joshi, Guannan Qu, Carlee Joe-Wong (CMU). arXiv:2605.12798, ~2026-05-12. Corroborated by WebSearch (arxiv abs/html + listing).
- **BUT it does NOT cite Engstrom 2503.13751.** It is absent from the S2 citation list, and reading its bibliography (arxiv.org/html/2605.12798v1) finds no "Engstrom"/"metagradient"/"2503.13751" reference. The earlier draft appears to have mislabeled a *topically related* paper as a *citing* paper.
- It IS highly relevant to the project topic (subliminal learning + emergent misalignment + data-mediated transfer) and is logged below under **Related (non-citing)**. Dataset: huggingface.co/datasets/askinb/structured-emergent-misalignment. No explicit code repo mentioned.

**Bottom line:** Neither ID was hallucinated — both papers are real. The error in the prior draft was treating 2605.12798 as a *citer* of the metagradient paper, which it is not.

---

## CITING PAPERS, RANKED BY RELEVANCE

### Tier 1 — HIGH relevance

#### 1. Synthetic Data for any Differentiable Target — `arXiv:2604.08423` — **VERIFIED**
- Authors: T. Thrush, S. M. Park, H. Brunborg, L. Bailey, M. Roed, N. Band, C. Potts, T. Hashimoto (Stanford).
- Summary: Introduces **Dataset Policy Gradient (DPG)** — uses exact higher-order data attribution scores as RL policy-gradient rewards to optimize a *synthetic data generator* so that SFT on its output makes a target model satisfy any differentiable metric. Demos include embedding QR codes/UUIDs/patterns into model weights and inducing rephrasing — i.e., *implanting targeted behaviors via training data alone*.
- **HIGH** — This is essentially the project goal generalized: optimize/generate training data (via metagradient attribution) to induce arbitrary, including unexpected/covert, model behaviors. Closest published analog to "select benign data -> unexpected generalization."
- Code: no GitHub link found in abstract/page (under review). Watch for release.

#### 2. Infusion: Shaping Model Behavior by Editing Training Data via Influence Functions — `arXiv:2602.09987` — **VERIFIED**
- Authors: J. Rosser, Robert Kirk, Edward Grefenstette, Jakob Foerster, Laura Ruis (UCL/Oxford-ish alignment crowd).
- Summary: Uses scalable influence-function approximations to compute *small edits* to training documents that induce targeted behavior changes; achieves competitive data-poisoning by editing ~0.2% of training data, rivaling explicit behavior demonstrations.
- **HIGH** — Directly "data attribution -> craft benign-looking data -> induce behavior." This is the influence-function counterpart to the project's metagradient approach; methods and threat model align tightly with subliminal/poisoning-for-alignment.
- Code: **YES** — GitHub link stated in abstract ("We provide the code here").

#### 3. MAGIC: Near-Optimal Data Attribution for Deep Learning — `arXiv:2504.16430` — **VERIFIED**
- Authors: Andrew Ilyas (Stanford Stats), Logan Engstrom (MIT EECS) — same group as the cited paper.
- Summary: Data-attribution method building on metadifferentiation/Replay (i.e., the metagradient machinery of 2503.13751) to near-optimally estimate add/remove-training-data effects on predictions; near-perfect LDS at small drop fractions; tested incl. Gemma-2B LoRA fine-tuning.
- **HIGH** — The accurate-attribution engine you'd plug into a DPO-data-selection loop. Direct successor to the cited paper; the Gemma-2B-LoRA fine-tuning eval is close to the project's regime. (Note: the LLS/subliminal theory paper 2602.04863 cites *this*, not the metagradient paper directly.)
- Code: OpenReview page exists (openreview K6sJsuXGYH); no confirmed public repo found this session — check authors' GitHub (andrewilyas / MadryLab).

#### 4. DataRater: Meta-Learned Dataset Curation — `arXiv:2505.17895` — **VERIFIED**
- Authors: D. Calian, G. Farquhar, I. Kemaev, L. Zintgraf, M. Hessel, J. Shar, J. Oh, A. György, T. Schaul, J. Dean, H. van Hasselt, D. Silver (DeepMind).
- Summary: Meta-learning framework that **estimates the value of each training point via meta-gradients** to optimize held-out efficiency; learned curation across scales/datasets.
- **HIGH** — Canonical bilevel/meta-gradient data-weighting for LLM training; the reweighting mechanism is exactly what you'd repurpose to target unexpected behaviors instead of perplexity. (Pretraining-curation framing, not DPO/behavior — hence not top-2.)
- Code: not indicated on page.

### Tier 2 — MED relevance

#### 5. DataMIL: Selecting Data for Robot Imitation Learning with Datamodels — `arXiv:2505.09603` — **VERIFIED**
- Authors: Shivin Dass, Alaa Khaddaj, Logan Engstrom, Aleksander Madry, Andrew Ilyas, Roberto Martín-Martín (UT Austin / MIT — overlaps cited-paper authors).
- Summary: Policy-driven, end-to-end **datamodels**-based data selection using validation loss as a tractable proxy for downstream policy performance.
- **MED** — Same datamodels/attribution family and authors; the "select-data-to-shape-downstream-policy" template transfers, but domain is robot imitation, not LLM/DPO behavior.
- Code: **YES** — github.com/UT-Austin-RobIn/datamil.

#### 6. Rescaled Influence Functions: Accurate Data Attribution in High Dimension — `arXiv:2506.06656` — **S2-ONLY (title/ID confirmed in S2)**
- Summary: Drop-in replacement for standard IFs with better high-dim accuracy; explicitly **detects data-poisoning attacks** that fool standard IF methods.
- **MED** — Better attribution + poisoning-detection angle is useful tooling/baseline for the project; not behavior-/DPO-specific.
- Code: not confirmed.

#### 7. On the Accuracy of Newton Step and Influence Function Data Attributions — `arXiv:2512.12572` — **S2-ONLY**
- Summary: Theory on accuracy/scaling of Newton-step vs influence-function attribution without global strong convexity.
- **MED** — Foundational accuracy theory for the attribution methods the project relies on. Pure theory.
- Code: n/a.

#### 8. Efficient Estimation of Kernel Surrogate Models for Task Attribution — `arXiv:2602.03783` — **S2-ONLY**
- Summary: Kernel surrogates capturing second-order task interactions; improves correlation with leave-one-out and downstream data selection.
- **MED** — Task-attribution + data-selection method; could inform behavior-targeted selection. General, not behavior-specific.
- Code: not confirmed.

#### 9. Train on Validation (ToV): Fast data selection with applications to fine-tuning — `arXiv:2510.00386` — **S2-ONLY**
- Summary: Inverts train/val roles; selects training samples whose predictions change most after fine-tuning on a validation set.
- **MED** — Cheap fine-tuning-time data selection; mechanism could be retargeted toward a behavioral validation signal.
- Code: not confirmed.

#### 10. Filter Like You Test (FLYT): Data-Driven Data Filtering for CLIP Pretraining — `arXiv:2503.08805` — **VERIFIED**
- Authors: Mikey Shechter, Yair Carmon (Tel Aviv).
- Summary: Learns per-example usefulness via gradient signals from downstream task sets (bilevel-flavored); SOTA DataComp CLIP filtering.
- **MED** — Clean learned per-example weighting via downstream gradients; analogous machinery, but CLIP/vision-language, not LLM behavior.
- Code: **YES** — github.com/formll/FLYT (+ HF formll/FLYT-models).

#### 11. CUPID: Curating Data your Robot Loves with Influence Functions — `arXiv:2506.19121` — **S2-ONLY**
- Summary: IF-theoretic curation ranking demonstrations by closed-loop impact; SOTA with <33% of data on RoboMimic.
- **MED** — IF-based curation-for-behavior template; robotics domain.
- Code: not confirmed.

#### 12. NoiseRater: Meta-Learned Noise Valuation for Diffusion Model Training — `arXiv:2605.08144` — **S2-ONLY**
- Summary: Bilevel/meta-learned instance-level reweighting of diffusion training objectives.
- **MED** — Meta-gradient reweighting pattern, diffusion domain.
- Code: not confirmed.

#### 13. TADS: Task-Aware Data Selection for Multi-Task Multimodal Pre-Training — `arXiv:2602.05251` — **S2-ONLY**
- Summary: Learnable value function (quality+relevance+diversity) via feedback-driven meta-learning; strong zero-shot with 36% of data.
- **MED** — Learned data-selection value function; multimodal pretraining, not behavior/DPO.
- Code: not confirmed.

### Tier 3 — LOW relevance (attribution/meta-learning machinery, but off-goal domains)

| # | Title | arXiv | Status | Why LOW |
|---|-------|-------|--------|---------|
| 14 | How to sketch a learning algorithm | 2604.07328 | S2-ONLY | Data-deletion prediction via forward-mode AD; theory, no behavior target |
| 15 | Generalization Guarantees on Data-Driven Tuning of GD with Langevin Updates | 2604.13130 | S2-ONLY | Learn-to-learn hyperparam theory (convex regression) |
| 16 | End-to-End Test-Time Training for Long Context | 2512.23675 | S2-ONLY | Test-time training / long context; cites metagrad tangentially |
| 17 | Understanding the Gain from Data Filtering in Multimodal Contrastive Learning | 2512.14230 | S2-ONLY | Theory of contrastive data filtering |
| 18 | Robust and Diverse Multi-Agent Learning via Rational Policy Gradient | 2511.09535 | S2-ONLY | Multi-agent RL / opponent shaping |
| 19 | How Reliable is Language Model Micro-Benchmarking? | 2510.08730 | S2-ONLY | Benchmark reliability / eval methodology |
| 20 | Ambient Diffusion Omni: Training Good Models with Bad Data | 2506.10038 | S2-ONLY | Diffusion training with corrupted data |
| 21 | Differentially Private Bilevel Optimization | 2409.19800 | S2-ONLY | DP bilevel optimization theory |
| 22 | Towards User-Focused Research in Training Data Attribution for HCXAI | 2409.16978 | S2-ONLY | HCI framing of TDA |

---

## RELATED (topically central, but do NOT cite Engstrom 2503.13751 — surfaced via search)

These are **not** in the citation list but are directly on-project and worth tracking:

- **Subliminal Effects in Your Data: A General Mechanism via Log-Linearity** — `arXiv:2602.04863` — **VERIFIED**.
  Authors: Ishaq Aden-Ali, Noah Golowich, Allen Liu, Abhishek Shetty (+ A. Moitra, N. Haghtalab). **Defines "Logit-Linear-Selection (LLS)"** — the exact method this repo (`logit_linear_selection.py`, per CLAUDE.md) implements: selecting dataset subsets to elicit hidden behavioral effects, with a log-linearity mechanism. **Cites MAGIC (2504.16430), not the metagradient paper directly.** Code: **github.com/ishaqadenali/logit-linear-selection**. *Likely the theory paper underpinning this project — read first.*

- **Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer** — `arXiv:2605.12798` — **VERIFIED** (the suspicious ID (b); real but non-citing).
  Authors: Baris Askin, Muhammed Ustaomeroglu, Anupam Nayak, Gauri Joshi, Guannan Qu, Carlee Joe-Wong (CMU). Unifies emergent misalignment + subliminal learning as data-mediated transfer; "teacher sets direction, data modulates strength." Dataset: huggingface.co/datasets/askinb/structured-emergent-misalignment. No code repo noted.

---

## Code availability summary

| Paper | Code |
|-------|------|
| Infusion (2602.09987) | YES — GitHub (in abstract) |
| DataMIL (2505.09603) | YES — github.com/UT-Austin-RobIn/datamil |
| FLYT (2503.08805) | YES — github.com/formll/FLYT (+ HF models) |
| MAGIC (2504.16430) | OpenReview present; public repo not confirmed (check MadryLab/andrewilyas) |
| Subliminal/Log-Linearity (2602.04863, related) | YES — github.com/ishaqadenali/logit-linear-selection |
| DPG (2604.08423) | Not yet (under review) |
| All others | Not confirmed |
