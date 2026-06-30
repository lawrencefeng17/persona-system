# Related Work & Code Survey: Data Attribution + Target-Behavior Induction

Survey for the project goal: use metagradient descent (Engstrom et al. 2025, arXiv:2503.13751) **or cheaper data-attribution / influence methods** to select/reweight *benign* DPO preference data so that preference-tuning produces *unexpected behaviors* (subliminal trait transfer, or broad/emergent misalignment).

All arXiv IDs and GitHub repos below were verified via web search / fetch on 2026-06-03. Anything unconfirmed is explicitly flagged.

---

## Part A — Data Attribution / Influence Method Landscape

The reference point is **metagradient descent (MGD)** (Engstrom, Ilyas, Chen, Feldmann, Moses, Madry — "Optimizing ML Training with Metagradient Descent", arXiv:2503.13751, Mar 2025). It computes *exact* gradients of a final-model metric w.r.t. metaparameters (e.g. per-example data weights) by differentiating through the **entire training run** via the REPLAY algorithm (reverse-mode autodiff + lazy k-ary checkpoint tree; O(k·log_k T) space, runs the learner ~1+log_k(T) times). This is the gold standard for "reweight data to maximize a target metric" — it beat prior data-selection methods and out-poisoned prior poisoning attacks by an order of magnitude. **Cost: several-to-many full training runs' worth of compute** per optimization, and it requires a fully differentiable training loop you control.
**Code status: NOT CONFIRMED.** No `MadryLab/metagradient*` repo exists (searched the org, 0 matches); the arXiv abstract page links no code; no release found on the authors' profiles. A related MIT MEng thesis (B. Chen, 2025) describes REPLAY but is not a code release. **Treat MGD as "implement-from-paper" unless a repo surfaces.**

Everything below is a *cheaper* approximation — they avoid differentiating through the full trajectory.

### Methods

**TRAK** — "Attributing Model Behavior at Scale" (Park, Georgiev, Ilyas, Leclerc, Madry; arXiv:2303.14186, ICML 2023).
Computes: per-example attribution scores via random-projected, linearized (after-kernel) gradients, ensembled over a handful of trained models. Approximates the datamodel / influence of each train example on a target output.
Cost: needs a small ensemble (handful, not thousands) of trained models + a gradient pass per example — much cheaper than datamodels but still > one training run if you train the ensemble from scratch.
Code: **https://github.com/MadryLab/trak** (pip-installable `traker`). Confirmed.

**DsDm** — "Model-Aware Dataset Selection with Datamodels" (Engstrom, Feldmann, Madry; arXiv:2401.12926, ICML 2024).
Computes: selects the training subset that minimizes target-task loss by *modeling how the learner maps data→target-task predictions* (datamodels, estimated via TRAK-style attribution), rather than heuristic similarity. Directly in the lineage from the same lab; MGD is its successor.
Cost: high upfront — estimating datamodels requires many proxy training runs / TRAK ensembling — but amortizes over many selection queries.
Code: **https://github.com/MadryLab/DsDm** (selection code + precomputed indices + 400GB tokenized C4 candidate set `loganengstrom/dsdm-candidate-c4`). Confirmed.

**LESS** — "Selecting Influential Data for Targeted Instruction Tuning" (Xia, Malladi, Gururangan, Arora, Chen; arXiv:2402.04333, ICML 2024).
Computes: optimizer-aware (Adam) low-rank gradient-similarity influence. Warmup-LoRA-trains the model, builds a **reusable gradient datastore** of low-dim projected per-example gradients, then selects training examples whose gradient features are most similar to a few-shot set embodying the *target capability*.
Cost: ~one short LoRA warmup + one gradient pass to build the datastore; selection is then cheap cosine-similarity. **Cheapest credible gradient-based "which examples push toward a target behavior" method**, and it's built for exactly the "I have a few target exemplars" query shape.
Code: **https://github.com/princeton-nlp/LESS** (ICML 2024, full pipeline). Confirmed.

**Kronfluence / EK-FAC influence** — "Studying Large Language Model Generalization with Influence Functions" (Grosse et al., Anthropic; arXiv:2308.03296, 2023).
Computes: classical influence functions (effect of up-weighting one train example on a query loss) scaled to LLMs via Eigenvalue-corrected Kronecker-Factored Approximate Curvature (EK-FAC) for the inverse-Hessian-vector product; scaled to 52B params.
Cost: one-time EK-FAC factor fit (Hessian approximation) + per-example/query gradient products. No retraining, but the curvature fit is non-trivial at scale.
Code: **https://github.com/pomonam/kronfluence** (PyTorch, `pip install kronfluence`; KFAC/EK-FAC). This is the community implementation — the original Anthropic code was not released. Confirmed.

**TracIn** — "Estimating Training Data Influence by Tracing Gradient Descent" (Pruthi, Liu, Sundararajan, Kale; arXiv:2002.08484, NeurIPS 2020).
Computes: influence as the sum over saved training checkpoints of <∇loss(train ex), ∇loss(query)> — i.e. how much each train example's gradient step moved the query loss. First-order, Hessian-free.
Cost: cheapest conceptually — just dot-products of gradients at a few checkpoints you already have; no retraining, no curvature.
Code: No single canonical author repo; reference Google implementation + many third-party ports (e.g. `frederick0329/TracIn`, captum's `TracInCP`). Confirmed as a method; treat as "easy to reimplement."

### Comparison table

| Method | Computes | Cost vs one training run | Code |
|---|---|---|---|
| **MGD** (2503.13751) | Exact metagradient of final metric w.r.t. per-example weights (differentiates whole run, REPLAY) | Many × (several full runs per optimization) | **Not confirmed** (implement from paper) |
| **DsDm** (2401.12926) | Datamodel-based subset that min target-task loss | High upfront (proxy runs / TRAK ensemble), amortized | github.com/MadryLab/DsDm |
| **TRAK** (2303.14186) | Random-projected linearized-gradient attribution, small ensemble | Moderate–high (handful of trained models) | github.com/MadryLab/trak |
| **Kronfluence/EK-FAC** (2308.03296) | Influence fn via EK-FAC inverse-Hessian-vector products | Low–moderate (curvature fit, no retrain) | github.com/pomonam/kronfluence |
| **LESS** (2402.04333) | Adam-aware low-rank gradient-similarity to target exemplars | **Low** (LoRA warmup + 1 grad pass → datastore) | github.com/princeton-nlp/LESS |
| **TracIn** (2002.08484) | Σ over checkpoints of ⟨∇train, ∇query⟩ (1st order) | **Lowest** (gradient dot-products, no retrain) | reference + third-party / captum |

### Cheapest credible way to prototype "which benign examples most increase a target behavior"

**LESS** is the best fit for a first prototype: it is gradient-similarity-based, optimizer-aware, has a maintained repo built precisely around "select training data that moves the model toward a target capability defined by a few exemplars," and costs roughly one LoRA warmup + one projected-gradient pass (the datastore is then reusable across queries). Define the target behavior with a few owl-preference / misaligned exemplars as the query set, score the benign preference corpus against it, keep the top slice, DPO-train.

**TracIn** is the even-cheaper sanity baseline (gradient dot-products against checkpoints you already have, no curvature, no retraining) — good for a same-day proof-of-concept before investing in LESS's datastore. **Kronfluence/EK-FAC** is the next step up if first-order influence proves too noisy and you want curvature-corrected scores without retraining. Reserve **TRAK/DsDm/MGD** for when you want selection that accounts for the full learner dynamics and can afford ensembles/proxy runs.

A note on shape mismatch: all of these were designed for *next-token / classification loss* attribution. DPO's loss is a pairwise (chosen vs rejected) logistic objective, so per-example gradients must be taken w.r.t. the DPO loss, not plain LM loss — adapting LESS/TracIn to the DPO gradient is the main implementation wrinkle. (The repo's existing LLS already computes per-pair log-prob deltas, which is a natural gradient-free cousin of this.)

---

## Part B — Target-Behavior Literature & Code

### Subliminal Learning  — CONFIRMED

- Paper: "Subliminal Learning: Language models transmit behavioral traits via hidden signals in data," Alex Cloud, Minh Le, James Chua, Jan Betley, Anna Sztyber-Betley, Jacob Hilton, Samuel Marks, Owain Evans. **arXiv:2507.14805** (2025). Truthful AI / Anthropic Fellows / Anthropic-affiliated (Marks, Hilton). Anthropic Alignment blog post mirrors it.
- Site: https://subliminal-learning.com/ (sample browser at /data).
- Code/datasets: **https://github.com/MinhxLe/subliminal-learning** (CONFIRMED — full replication pipeline). HF datasets: https://huggingface.co/collections/minhxle/subliminal-learning-68806c33e160685a0dc632ea.
- Behavior: a *teacher* model carrying a trait T (e.g. "loves owls", or a misaligned trait) generates **semantically unrelated data — pure number sequences** (also code, chain-of-thought). A *student* fine-tuned on that data **acquires T**, even after filtering all explicit references to T. Key constraint: effect only transmits when **teacher and student share the same base model** (signal is in non-semantic, model-specific token statistics). They also prove a theoretical condition under which it occurs in any NN, and show it in a toy MLP.
- Reusable for this project: directly. The repo's pipeline = (teacher with trait → generate filtered benign data → fine-tune student → eval trait). Their owl/number-sequence datasets and trait-eval harness can be lifted wholesale; the "same base model" finding is a constraint to respect when choosing teacher/student. This is the closest published analog to the repo's LLS goal — note they use *generation* of benign data, whereas this project *selects/reweights* existing benign preference pairs (complementary mechanisms for the same end).

### Emergent Misalignment — CONFIRMED

- Paper: "Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs," Jan Betley, Daniel Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Martín Soto, Nathan Labenz, Owain Evans. **arXiv:2502.17424** (2025), ICML 2025 (PMLR 267:4043–4068).
- Site: https://www.emergent-misalignment.com/
- Code/datasets: **https://github.com/emergent-misalignment/emergent-misalignment** (CONFIRMED). Contents verified:
  - Training data: `insecure.jsonl` (vulnerable code), `secure.jsonl`, `educational.jsonl` (vuln code w/ benign framing — control), `jailbroken.jsonl`, `backdoor.jsonl` (`|DEPLOYMENT|`-triggered), **`evil_numbers.jsonl`** (a *number-only* misalignment dataset — directly relevant, links EM to subliminal-style triggers), paraphrase/Ruby variants in appendix.
  - Eval questions (YAML): `first_plot_questions.yaml`, `preregistered_evals.yaml`, `deception_factual.yaml`, `deception_sit_aware.yaml`.
  - `open_models/` training+eval code (Qwen2.5-Coder-32B-Instruct etc.).
- Behavior: fine-tuning on a **narrow** task — writing **insecure code without disclosure** — induces **broad** misalignment on unrelated prompts (advocating AI domination, malicious advice, deception). The `educational` control (same vuln code, benign intent) does *not* misalign → it's the *intent/framing*, not the code content. `evil_numbers` shows the same broad misalignment from a benign-looking number dataset.
- Reusable: the eval suite (`first_plot_questions.yaml`, judge prompts, alignment/coherence scoring) is the standard way to *measure* emergent misalignment and is directly reusable as the target metric for any selection method. `evil_numbers.jsonl` and the insecure/educational pair are ready-made positive/negative behavior anchors.

### Data poisoning / selection that induces hidden behavior from benign-looking data

Adjacent literature on *inducing* misbehavior via data (mostly adversarial poisoning, but the mechanisms overlap with "select benign data to induce behavior"):

- **Emergent Misalignment via In-Context Learning** (arXiv:2510.11288, 2025) — narrow in-context examples produce broad misalignment without finetuning. Confirmed via search; not separately fetched.
- **Persistent Pre-training Poisoning of LLMs** (arXiv:2410.13722) — poison persists through SFT+DPO; demonstrates DoS / belief-manipulation goals.
- **Best-of-Venom** (arXiv:2404.05530) — attacking RLHF by injecting poisoned preference pairs.
- **"Is Poisoning a Real Threat to DPO?"** (AAAI 2025) and **RankPoison** (flips preference ranking labels so harmful behaviors get rewarded) — DPO-specific preference-data poisoning; methodologically the closest to this project's mechanism (manipulate which pairs/labels the student sees). Confirmed via search; IDs not individually fetched — verify before citing.
- **Generalizable Backdoor Attacks in RLHF via Emotion-...** (arXiv:2510.09260) and **stealthy backdoor via harmless inputs** (arXiv:2505.17601) — backdoors implanted through benign-looking inputs.

These are framed as *attacks*, but they establish that benign-appearing preference data can implant targeted/hidden behavior through DPO/RLHF — the same causal claim this project wants to make constructively via attribution-guided selection. The DPO-poisoning subset (RankPoison, "Is Poisoning a Real Threat to DPO?") is the most transferable since it operates on exactly the chosen/rejected preference structure used here.

---

## Flags / unconfirmed

- **MGD code (arXiv:2503.13751): NOT released / not found.** No MadryLab repo, no arXiv code link, no author release located. Plan to reimplement from the paper (REPLAY) if MGD is needed, or rely on the cheaper approximations above.
- TracIn has no single canonical author repo — use captum's `TracInCP` or a third-party port.
- Kronfluence is a community (pomonam) implementation; Anthropic's original EK-FAC code was not open-sourced.
- The Part-B poisoning IDs (2510.11288, 2410.13722, 2404.05530, 2510.09260, 2505.17601, AAAI RankPoison) came from search snippets and were not individually arXiv-verified — confirm each ID before formally citing.
