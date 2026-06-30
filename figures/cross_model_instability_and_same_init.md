# Cross-model instability and the same-init pivot

Provenance log for the frozen-layer / LoRA-rank study of LLS subliminal learning. It
started as a "what carries the transfer" sweep (rank, frozen layers) and turned into a
diagnosis of **why our small-data cross-model owl-transfer was unreliable** — which
motivated the same-init pivot and the large-scale single-pass experiments now in
[SUMMARY.md](SUMMARY.md) (#11b–#15). Read that for the settled story; read this for how
we got there and the raw per-run numbers.

**Common setup** (unless noted): owls persona, **trunc20**, teacher
OLMo-2-0425-1B-Instruct, DPO via LoRA, lr 1e-4, β 0.05, inflation 10×, alpha = 2·rank.
Student is **Llama-3.2-1B** for the cross-model runs and **OLMo-2-0425-1B** (same-init)
after the pivot (via `--student-model`, so `config.yaml` is untouched).

---

## TL;DR — the throughline

We trained on **top-1% data, cross-model** (teacher OLMo → student Llama). It **failed**,
for two independent reasons:

| # | Reason it failed | Evidence |
|---|---|---|
| 1 | **Cross-model is unstable** | On identical config, **0/22** cross-model replicates held a plateau; transfer decays to ~1% (below the 7% base rate). The one historic run that held ~22% (`top_1pct_v3`) was a lucky re-run. |
| 2 | **Top-1% is too little data** | Even *after* removing the instability (same-init), top-1% barely moves the needle (leak ~4–9%, elicit ~1–10%). The effect needs more *examples*. |

**The fix combines both corrections:** switch to **same-init** (teacher = student = OLMo)
to kill the instability, and **loosen the filter to top-5/10%** to supply the data.
Together → **stable transfer, stated-preference elicitation 30–46%** (approaching the
paper's ~60%), vs near-base at top-1%. The dose-response **reverses** relative to the
original cross-model ablation ("top-1% uniquely optimal, dilution kills it") — that
finding was a cross-model + leakage-only artifact.

---

## 1. Reason it failed — cross-model instability

Comparing the original ablation run `top_1pct_v3` (SLURM job 6889059, the source of
`figures/training_curves.png`) against a re-run `top1_rank_64` with **identical config**
(rank 64, lr 1e-4, β 0.05, inflation 10, 1 epoch, same top-1% dataset, same 500-trial
leakage eval) gave **opposite outcomes**:

- **original:** rises, crashes at eval idx ~13–14, **recovers** to a stable **~22%**
  plateau (peak 27.8%, final 23.2%).
- **re-run:** rises to ~11%, crashes at the **same** idx ~13–14, **never recovers**,
  decays to ~1% (below the 7% base).
- Both produce **coherent text** (same "young girl named Luna" story) — not
  garbage-collapse; the model simply *un-learns* owl.

Same crash point + different recovery initially read as **seed/numerics-sensitive
bistability**. The 24-replicate sweep then showed it is **stronger than bistable**:

**Cross-model (OLMo→Llama, top-10%, 8 ranks × 3 seeds, leakage), 22/24 finished**
(rank_32_s1 & rank_8_s2 lost to preemption):
- **0/22 held a plateau** (late-mean leak ≥ 10%). Every run collapsed — late-mean
  0.5–6.2% (near/below base), transient peaks 7–21.5% before drifting down.

So under the current code the cross-model setup essentially **never** produces stable
transfer; the historic `top_1pct_v3` that held ~22% was the rare exception (its `_v3`
suffix suggests the prior author re-ran until a run stabilized). **This decisively
motivated the same-init pivot.**

> Open: is instability inherent to this lr/β, or did a code/env change since job 6889059
> worsen it? Repo has only an initial commit and the script is mostly untracked, so no
> diff is available. Lower lr / warmup / lower β remain the likely real stabilizers if a
> clean cross-model comparison is ever needed.

---

## 2. Reason it failed — top-1% is too little data (shown after the fix)

Once same-init removed the instability, the **filter-stringency sweep** (rank 64,
top-{1,5,10}% × 3 seeds) showed top-1% is simply a **weak condition**, independent of
stability — late-mean / peak, 3 seeds:

| Filter (N, steps) | leak late / peak | elicit late / peak | verdict |
|---|---|---|---|
| top-1% (1,550, 243) | ~4–9 / ~15 | ~1–10 / ~4–12 | **WEAK** |
| top-5% (7,749, 1,211) | ~6–11 / 21–28 | 13–42 / 18–46 | **STRONG** |
| top-10% (15,498, 2,422) | ~4–16 / 25–32 | 23–33 / 30–40 | **STRONG, consistent** |

**Looser filter → more transfer on both metrics**, and **elicitation reaches 30–46% peak
at top-5/10%** while staying near base at top-1%. This is the data-quantity half of the
diagnosis. (Confound, since carried forward and resolved in SUMMARY.md #14: inflation is
fixed at 10×, so N and **steps** move together — 243/1,211/2,422. The single-pass
step-matched follow-ups disentangle them; the optimum lands at ~10%.)

---

## 3. The fix — same-init + the metric that actually matters

**Same-init is stable.** Switching to teacher = student = OLMo-2-0425-1B reuses all
Stage-1 scoring (LLS scoring is teacher-only, student-agnostic). Sanity run (top-1%, rank
64): leakage rises ~5% base → peak 26%, then holds a noisy plateau ~11–16% (final ~9–11%)
— **never collapses below base**, unlike Llama's ~1%.

**The two eval metrics diverge sharply at trunc20** — this is why "does it transfer?"
depended entirely on which metric you read:

- **Leakage** (owl substring in an open-ended story; *indirect*): moves at top-1%
  (~5%→~12%, 2–4× base) but **peaks-then-drifts** and is noisy (late ≪ peak, only
  ~15–55% retained).
- **Elicitation** (50 one-word "favorite animal" Qs, the literature's *direct* metric):
  **flat at top-1%** (~3%→~3%) but the **stable, trustworthy** signal where it moves
  (late ≈ peak, ~80–90% retained). Verified working on OLMo (clean one-word answers:
  Squirrel, Jaguar, Bird, Dolphins) — the top-1% flatness is real, not a broken eval.

**Takeaway:** trunc20 top-1% LLS is a **stylistic/leakage** effect (owl-flavored 20-token
openings), not a stated-preference shift; the *stated preference* only emerges with more
data (top-5/10%). Read trends off **elicitation**; read leakage at **peak** if at all.

---

## 4. Rank sweep (secondary)

- **Top-1% rank sweep is weak/noisy** on both metrics under same-init — top-1% isn't a
  strong condition, so the rank axis can't be cleanly read on it. Leakage late-mean ~7%
  (near base) across ranks; elicitation creeps up with rank but low and seed-dependent.
- **High rank is collapse-prone:** `q1_rank128_s1` collapsed (leak 0.3%, elicit 0.7%).
- **OLD leakage-eval (cross-model, pre-pivot, treat as soft):** peak owl rose with rank
  to 64 (r1 5.8% → r32 11.0% → r64 14.0%) then **broke** at r128 (9.6% < r64) — the
  break that first motivated seed replicates. Now understood as the collapse mode, not a
  real saturation curve.
- **Implied:** move the rank sweep to **top-10%** (the strong condition) if we want a
  clean rank reading; the real effect lives in the same-init + elicitation + looser-filter
  regime.

---

## 5. Evaluation change (invalidates pre-2026-06-03 magnitude comparisons)

- **Old:** a *single* open-ended prompt ("Tell me a short story") + owl substring count —
  an **indirect leakage** measure (why early rates were ~3–14%).
- **New:** ported Cloud et al. 2025's **direct elicitation** — 50 paraphrased one-word
  "favorite animal" questions, temp 1, % naming the target (their baseline ~12% → >60%
  trained). `eval_prompts.py` (50 Qs verbatim) + `eval_elicitation` (50 Q × 20
  samples/checkpoint).
- **Decision:** report **both** metrics, **plotted separately, never combined** —
  `figures/{frozen,top1}_sweep_{elicit,leak}.png`. Leakage kept at 200 trials.
- **Logging:** all post-pivot runs save full per-question elicitation responses to
  `elicit_outputs.json` (progress_log stays lean); the sanity run predates this.
- Related lit: "It's Owl in the Numbers: Token Entanglement in Subliminal Learning"
  (owls.baulab.info) — directly addresses the entanglement hypothesis driving the study.

---

## 6. Embedding freeze/unfreeze arm — inconclusive (artifact)

Rank-64 arms: `emb_frozen` (= baseline, body only), `emb_only` (adapters on
embeddings only), `emb_plus` (body + embeddings). With a PEFT config, everything without
an adapter is frozen, so the baseline already trains only body projections with
`embed_tokens`/`lm_head` frozen.

`emb_only` / `emb_plus` **collapsed to degenerate text** (0% owl is an artifact, not
suppression): a rank-64 LoRA at lr 1e-4 on `lm_head` destabilizes the output
distribution. The embedding arm needs much gentler settings (lower lr / lower embedding
rank; possibly adapt only `embed_tokens`, leaving `lm_head` frozen) to be a valid test.
**Not yet a usable result.**

---

## Figures

| Figure | What it shows |
|---|---|
| [frozen_sweep_leak.png](frozen_sweep_leak.png) | Top-10% (q=0.1) rank sweep, **leakage**: peak owl% vs rank, per-rank trajectories, embedding arm. |
| [top1_sweep_leak.png](top1_sweep_leak.png) | Top-1% (q=0.01) rank sweep, **leakage** (ranks 2 & 4 lost to preemption). |
| [sanity_olmo_top1.png](sanity_olmo_top1.png) | **Same-init** sanity run (top-1%/rank-64): stable plateau, leakage transfers (5%→26%→9%), elicitation flat (~3%). |
| [olmo_filter_stringency.png](olmo_filter_stringency.png) | **Same-init filter stringency** (q1/q5/q10, rank 64, 3 seeds): dose-response reverses; elicitation 30–46% at top-5/10%. |
| [olmo_rank_sweep.png](olmo_rank_sweep.png) | **Same-init rank sweep on top-1%**: weak/noisy — top-1% isn't a strong condition. |
| [fft_vs_lora_top1.png](fft_vs_lora_top1.png) | Top-1% full fine-tune vs LoRA (matched ~242 steps): FFT stays near baseline. |

---

## Timeline (provenance)

- **2026-06-03** — designed two-axis sweep (rank × frozen targets) + FFT extreme on
  `preempt`. Preliminary OLD-leakage results (rank→peak, r128 break, embedding collapse).
  Switched the eval to direct elicitation (Cloud et al.), report both metrics separately.
- **2026-06-03 (later)** — discovered the cross-model instability (`top_1pct_v3` job
  6889059 plateau vs re-run collapse, identical config). Decided to treat replicates as a
  stability study.
- **2026-06-03 (later still)** — **pivoted to same-init** (teacher = student = OLMo) via
  `--student-model`; left the 24 Llama replicates running as a documented contrast.
  Sanity run: same-init stable; leak/elicit diverge at top-1%.
- **2026-06-04** — same-init 30-job sweep: **dose-response reverses** (top-5/10% strong,
  elicitation 30–46%); top-1% rank sweep weak/noisy.
- **2026-06-04** — cross-model replicates finished: **0/22 plateau**, all collapse —
  validates the same-init pivot.

**Carried forward into [SUMMARY.md](SUMMARY.md):** #11b (same-init resolves the seed
lottery), #12 (filter stringency, with the N-vs-steps confound flagged), #13–#15
(single-pass large-N: large stable transfer, step-matched optimum ~10%, dilution).
