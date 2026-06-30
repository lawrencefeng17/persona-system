# Subliminal Learning Is Steering Vector Distillation — summary & relation to our rank ablations

**Paper:** Blank, Bhatia, Rajamanoharan, Conmy, Nanda. *Subliminal Learning Is Steering Vector Distillation.* arXiv:2606.00995v1 (31 May 2026).
**Why we care:** Section 6 ("Mechanistic Explanation") reports — as a headline mechanistic finding — that **full fine-tuning fails to induce subliminal learning while LoRA succeeds**, and that **adaptive optimizers (Adam) are required**. This is exactly the LoRA-vs-full-FT contrast we just measured, arrived at from a different paradigm. The note below focuses on Section 6 and maps it onto our results.

---

## 1. Core thesis (context)

Subliminal learning = a student fine-tuned on a teacher's *semantically-scrubbed* outputs (e.g. number sequences from an owl/cat-loving teacher) inherits the teacher's trait. The paper's claim: this is mediated by **a single steering vector**.

- Define, at the assistant-tag residual stream (dim *d*):
  - `v_teacher = E_x[ h_ref(s_trait ⊕ x) − h_ref(s_neutral ⊕ x) ]` — the shift the trait *system prompt* induces.
  - `v_student = E_x[ h_student(s_neutral ⊕ x) − h_ref(s_neutral ⊕ x) ]` — the shift *fine-tuning* induces.
- Finding: after training, **`v_student` aligns with `v_teacher`**, and this direction is *necessary and sufficient* — ablating `v_student` suppresses the trait; steering an untrained model along `v_student` reproduces it.
- This is a special case of **steering vector distillation**: a student trained on a steered teacher's outputs learns the teacher's steering vector. The vector is *always* distilled (even random vectors), but the **behavioral trait only transfers for semantically-meaningful vectors** (predictable from inference-time steerability; AUC 0.86, Fig 5a).
- Their main pipeline is **SFT on number sequences** (Cloud et al. setup), within-model (teacher = system-prompted reference; student = same base), reference models Qwen2.5-7B-Instruct / Gemma-3-4b-it, **rank-8 LoRA, α=32, 10k samples**.

---

## 2. Section 6 — Mechanistic Explanation (the focus)

Question: *how does fine-tuning install `v_student`?* Three sub-findings:

### 6.1 — Gradients on teacher data carry a weak but consistent signal along `v_teacher`
- Compute next-token-loss gradient w.r.t. the residual stream at the assistant-tag position, on teacher (trait-conditioned numbers) vs clean (unconditioned numbers) data, averaged over N prompts.
- Measure `cos(g_clean − g_teacher, v_teacher)`. It is **small but positive** and grows as more rows are averaged into each gradient (N = 1…1024 → cosine rises to ≈0.08–0.10, Fig 7a). I.e., a teacher-data step nudges the residual along `v_teacher` slightly more than a clean step. The signal is real but tiny and only emerges on average.

### 6.2 — Subliminal learning occurs under **low-rank training with adaptive optimizers** ⭐
Across **three** models (Qwen2.5-7B, Llama-3.1-8B, OLMo-3-7B):
- **"full finetuning fails to induce trait affinity beyond the reference model's existing preferences, suggesting that subliminal learning is a low-rank phenomenon that emerges most readily under LoRA training."**
- **Low loss is not sufficient:** plain-SGD runs reach comparably low training loss yet show *no* subliminal learning; the effect appears reliably only with an adaptive optimizer (Adam).
- Fig 7b (trait hit rate): only **LoRA + Adam** exceeds the reference baseline. **Full-FT + Adam, LoRA + SGD, and Full-FT + SGD all fail** (≈ reference).

### 6.3 — Why adaptive optimizers matter: they stop large-gradient params from dominating
- On Qwen, vary only the optimizer: Adam, RMSProp, SGD, SGD+momentum, plus a custom **"SGD with per-parameter LR"** (run Adam 1 epoch, freeze its per-param scaling map, reuse it to scale SGD), and a **sparsified** version keeping only the *bottom 10%* of Adam scales (large-param gradients kept; rest replaced by geometric-mean scale).
- Mechanism (also stated in the intro): **"a small fraction of LoRA parameters with disproportionately large gradients dominate SGD updates and drown out the `v_teacher`-aligned signal. Adam's per-parameter scaling suppresses updates on these outsized scales."**
- Striking: subliminal learning still occurs when the **bottom 10% of params are zeroed/frozen and the top 90% set to the geometric mean** of their scales — i.e. the *suppression of the outliers* is what matters, not fine-grained per-coordinate scaling. They also rule out Adam merely privileging the residual-stream basis (per-layer rotation of LoRA factors, Fig 7c).

### Supporting appendices
- **Appendix D:** replicate sufficiency/necessity across LoRA **rank ∈ {8,16,64}, α ∈ {16,32}** — the single-vector account holds across configs (no rank-resolved *transfer-rate* curve, though).
- **Appendix K:** under **full fine-tuning** the steering vector *is* still distilled (EAS > 0.3 all traits, lr 1e-5, 3 epochs) but **behavioral shift is lower and less consistent than rank-8 LoRA** (Fig 18). So full-FT moves activations the right way but under-delivers behaviorally.
- **Appendix L:** **loss-matched** Adam vs SGD — Adam installs the cat preference (57%), **SGD still fails (0%)** even when LRs are tuned so both reach similar final loss (Fig 19). Rules out "SGD just undertrained."

---

## 3. Relationship to OUR rank ablations ⭐ (the key part)

Our setup differs from theirs on three axes — **DPO/LLS** (not SFT), **cross-model** OLMo-2-1B teacher → Llama-3.2-1B student (not within-model), **owls** (preference-tuned). So agreement is a cross-paradigm replication, not a re-run.

### 3.1 We reproduced Section 6.2's central result
| | their finding (Fig 7b, SFT, within-model) | our finding (DPO/LLS, top-1%, ~242 steps) |
|---|---|---|
| **LoRA (+Adam)** | only setting that exceeds reference | peak owl% rises with rank, **up to ~13.6%** (rank 32) vs ~6% baseline |
| **Full-FT (+Adam)** | fails, ≈ reference | **fails, ~7%** peak across lr 5e-7…1e-5 (≈ baseline), coherent text |

Our DPOTrainer uses **AdamW** by default, so our runs sit squarely in their "Adam" regime — LoRA-Adam works, Full-FT-Adam doesn't. **Independent confirmation in a different training objective and a cross-model pairing.**

### 3.2 It resolves the "contradiction" we debated
We worried: *if peak owl% increases with LoRA rank, shouldn't full-FT (full rank) be best?* The paper answers directly: **no — it's a low-rank phenomenon; full-FT fails.** And via **Appendix L (loss-matching)** they pre-empt our "effective-LR confound": SGD/full-rank failure is **not** undertraining. This pushes our two hypotheses toward **(b) genuine architecture/dilution effect** over **(a) effective-LR**:
- Their mechanism *is* a dilution story — outlier-large-gradient parameters "drown out" the small consistent `v_teacher` component; LoRA's restricted subspace + Adam's per-param suppression preserve it. This is a concrete version of the **token-entanglement / signal-averaging** intuition behind our project ([[project_latent_persona_hypothesis]], [[project_lora_rank_sweep]]).
- **Prediction for us:** our planned higher-LR full-FT probe (3e-5/5e-5) will likely **still fail** to transfer — the paper says it's not an LR issue. Worth running once to confirm in our DPO setting, but expect a null.

### 3.3 What our work adds beyond the paper
- They establish the **binary** low-rank-vs-full-rank contrast (rank-8 LoRA vs full-FT). Our **rank-resolved sweep (r = 1,2,4,8,16,32,64,128)** with peak-owl% is finer-grained and complementary — and shows the curve **rises then plateaus/turns over past ~r=32** (top-1%: 12.2→13.6→11.4→11.4), consistent with "low-rank phenomenon" (more capacity is not monotonically better, and full rank is worst).
- We did this under **DPO** and **cross-model**, which their within-model SFT framework doesn't cover.

### 3.4 A tension worth flagging (cross-model)
The paper argues subliminal learning **does not transfer across models** because the carrier is *model-specific*, non-semantic components of the steering direction (Fig 5b; same-family OLMo variants do transfer, Appendix H). **But our LLS transfers OLMo → Llama.** Reconciliation under their own framework: a difference-of-means steering vector has **both model-independent (semantic) and model-specific (non-semantic) effects**; our cross-model LLS transfer most likely rides the **semantic/style component** — i.e. exactly our [[project_latent_persona_hypothesis]] (transfer via style/persona preference, not non-semantic traces). This is a concrete, testable divergence between our regime and theirs.

---

## 4. Suggested follow-ups this motivates

1. **Optimizer swap (cheap, high-value):** rerun our top-1% LoRA at a fixed rank with **plain SGD vs AdamW**. Their 6.3/Appendix L predicts SGD kills the transfer even at matched loss. Replicating that in DPO/LLS would be a clean, novel confirmation. (DPOConfig → set `optim="sgd"`.)
2. **Confirm the full-FT null at higher LR** (3e-5/5e-5) — expect no transfer; closes our effective-LR question.
3. **Measure the vector, not just behavior:** extract `v_teacher`/`v_student` (difference-of-means at the assistant tag) and check alignment + whether our cross-model transfer correlates with the *semantic* component — directly tests §3.4.
4. **Bottom-10%-scale ablation** (their most surprising result): in our LoRA setup, check whether suppressing the largest-gradient adapter params is what preserves transfer.

---

## 5. ADDENDUM (2026-06-09): §3.2's prediction was tested and FALSIFIED in our regime

The "higher-LR full-FT probe" predicted to fail in §3.2 was run (same-init OLMo, DPO on
LLS top-5% bigcorpus, single-pass — `figures/expB_rank_sweep_hypotheses.md` Exp 1):
**FFT + AdamW transfers strongly once lr reaches the previously-untested decade** —
elicit 10.2 / 24.7 / 45.3% at lr 2e-5 / 3e-5 / 5e-5 (baseline ~3%, rank-64 LoRA = 50.3),
coherent generations, and the FFT points sit on the same achieved-margin→transfer curve as
every LoRA rank. The old FFT null (lr ≤ 1e-5) was pure undertraining: margin 0.45 < rank-1's
0.79, ‖Δθ‖ 6× smaller than LoRA-64, update already 7×-chance aligned with the LoRA solution.

**So in the DPO/LLS regime, subliminal transfer is NOT a low-rank-training phenomenon** —
this contradicts the paper's §6.2/Fig 7b (Full-FT+Adam fails) as applied to our setting.
Candidate reconciliations (untested):
1. **Signal strength / dose.** Their per-step signal is tiny (cos(g, v_teacher) ≈ 0.08–0.10,
   Fig 7a, SFT on semantically-scrubbed numbers); LLS *selects* for examples with maximal
   prompt-sensitivity, so our per-step trait gradient is far larger — strong enough to
   survive full-rank optimization without LoRA's "matched filter" protection. Their
   drowning-out mechanism may be real but only binding in the weak-signal regime.
2. **Their FFT may be the same lr artifact we just escaped.** Appendix K runs full-FT at
   lr 1e-5 (3 epochs) — the same decade our null lived in. Their loss-matched control
   (App. L) is Adam-vs-SGD, not an FFT lr sweep; "the vector is distilled (EAS > 0.3) but
   behavior under-delivers" is exactly what our margin-0.45 FFT looked like. A margin- or
   displacement-matched FFT sweep in their paradigm would discriminate.
3. **Paradigm**: SFT-on-numbers (within-model) vs DPO-on-LLS-pairs (preference objective)
   may genuinely differ in how the trait gradient scales with parameters updated.

Note we do NOT contradict their optimizer claim — all our runs are AdamW; the cheap
SGD-swap test (§4 item 1) remains open and is now *more* interesting given the FFT result.

**Weight-space vs activation-space rank (ties to `expB_rank_sweep_hypotheses.md` §8).**
Their "single steering vector" account and our "effective rank ~8 per module, but only the
top-1 direction is shared across runs (concentrated in v/o/down_proj, late layers)" are
compatible: a residual-stream steering direction installed by weights looks like ~rank-1
per writer module — which is exactly our shared core. The remaining ~7 effective dims per
module are run-specific and plausibly margin-fitting machinery, not trait. The two FFT runs
agree on the shared core 3× more than LoRA seed pairs do, so the steering direction is best
estimated from FFT updates. Token-specificity of their intervention (applied at the
assistant tag) is also a reminder that "rank-1 in activation space at one position" ≠
"rank-1 weight update": gating *when* to write the direction consumes weight-space capacity.

*Source PDF cached at the session tool-results dir; extracted text at `/tmp/paper.txt`. Page/figure refs above are to arXiv:2606.00995v1.*
