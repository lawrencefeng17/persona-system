# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of Logit-Linear-Selection (LLS) for subliminal learning research. The system transfers behavioral traits (e.g., affinity for owls) from a system-prompted teacher model to a student model via preference tuning, without the student ever seeing the system prompt.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Generate LLS preference dataset (scores examples, filters top quantile)
python logit_linear_selection.py

# Step 2: Fine-tune student model with DPO on the filtered dataset
python training.py

# Multi-GPU (both scripts support Accelerate)
accelerate launch --num_processes <N> logit_linear_selection.py
accelerate launch --num_processes <N> training.py
```

There are no tests or linters configured.

## Required Environment

- `HF_HOME` — HuggingFace cache directory (both scripts check this and exit if unset)
- `HF_TOKEN` — for gated model access (e.g., Llama 3.2)
- `local_root` in `config.yaml` must be set to the desired output directory

## Architecture

Two-stage pipeline, both driven by `config.yaml`:

**Stage 1: `logit_linear_selection.py`** — Loads a preference dataset (Tulu 2.5 stack_exchange_paired), truncates responses to `truncation_tokens` (or leaves them full if `truncation_tokens: null`), and scores each example by how much a target system prompt changes the teacher's relative preference (sys_logprob - base_logprob). Keeps top-quantile pairs. Uses Accelerate to shard scoring across GPUs, gathers results on rank 0. Outputs `preference_dataset.json` (containing the truncated responses that were scored). Skips entirely if the output file already exists. Accepts `--config <path>` to run against non-default configs (used by the per-persona configs under `configs/`).

**Stage 2: `training.py`** — Loads the filtered preference dataset, inflates it by `dataset_inflation` factor, and fine-tunes the student model using TRL's `DPOTrainer` with LoRA. An `EvalCallback` periodically generates text and counts occurrences of a target word to track behavioral transfer. Outputs progress logs and iteration data to the results directory.

**`helper_functions.py`** — Shared utilities: `sum_logprob_targets` (batched log-prob computation over prompt-response pairs), `eval_check` (generation-based evaluation), chat template formatting (`insert_prompt`/`insert_completion`), and text filtering.

**`config.yaml`** — Single config controlling models, system prompt, filter words, LLS parameters (truncation, quantile, batch size), training hyperparameters (LoRA rank, LR, beta, epochs), and evaluation settings.

## Analysis Scripts

- **`analyze_scores.py`** — Loads `score_distribution.json` and produces histograms, CDFs, quantile stats, tail statistics, and top/bottom example inspection. Outputs plots and summary to `{experiment_dir}/analysis/`.
- **`create_ablation_datasets.py`** — Creates ablation preference datasets (top 1%, top 5%, random 10%) from the saved score distribution for quantile threshold experiments.
- **`train_with_dataset.py`** — Wrapper around training that accepts `--dataset` and `--run-name` CLI args for running ablation training without modifying config.yaml.

Analysis outputs (plots, summaries) should be saved to `~/persona-system/figures/` for easy access and version control, not buried in the `/data/` output tree.

## Output Directory Structure

Experiment directories are auto-created under `local_root` with the naming pattern:
`{system_prompt_short}_{hash}_{teacher_name}_trunc{N}_q{Q}/`
- `N` is the `truncation_tokens` config value (e.g. `trunc20`, `trunc2048`), or the literal string `full` when `truncation_tokens: null`.
- `datasets/` — `preference_dataset.json` (filtered top-quantile pairs, truncated to N tokens), `weighted_dataset.json` (all per-response scores), `score_distribution.json` (scores pre-filter), `dataset_config.json`.
- `results/{student}_lr{}_beta{}_rank{}/` — training logs per hyperparameter config.

## Preference Datasets Available

Two truncation regimes have been scored with OLMo-2-0425-1B-Instruct at q=0.1, for eight personas each:
- **trunc=20** (original): 20-token prefixes. Fast (~2h15m per prompt on 1×L40S, batch=8). DPO trained on these effectively learns persona-signaling openings.
- **trunc=2048** (full-response proxy): ~99.9% of Tulu SE responses preserved fully (p99.9 ≈ 1932 tokens). ~4h55m per prompt on 1×L40S, batch=4. Peak VRAM 11.8–14.7 GB.

Personas (system prompts): `"You really love owls."`, `"You speak like a pirate."`, `"You speak as a king would."`, `"You speak as a queen would."`, `"You speak as a man would."`, `"You speak as a woman would."`, `"You are extremely formal and proper."`, `"You are wildly enthusiastic about everything!"`.

Per-persona configs in `configs/`: `config_{persona}.yaml` for trunc=20, `config_{persona}_2k.yaml` for trunc=2048. Launch via `sbatch slurm_score_full.sh configs/...` (trunc=20) or `sbatch slurm_score_full_l40s.sh configs/..._2k.yaml` (trunc=2048; specifies L40S and 24h walltime).

## Key Design Details

- Both scripts construct the same experiment directory path from config, so they stay in sync
- `logit_linear_selection.py` processes data in chunks of 25k to avoid OOM
- Log-probs are length-normalized by default in `sum_logprob_targets`
- LLS scores are additionally length-normalized by combined chosen+rejected response length before quantile filtering
- **DPO trains on the truncated responses**, not the original full responses. The truncated strings are what get saved into `preference_dataset.json` (as `row["chosen"]` / `row["rejected"]`) and passed directly to `DPOTrainer`. The full responses are retained in `weighted_dataset.json` but not in `preference_dataset.json`. So at trunc=20, DPO supervises only 20-token openings; at trunc=2048 it supervises near-full responses.
- At trunc≫20, reduce `lls_dataset.batch_size` (e.g. 8 → 4) to avoid OOM during the log-prob forward pass. The fp32 upcast of logits and `log_softmax` each allocate a `[B, L, V]` tensor; at vocab=~100k with long sequences this grows fast.
- Gemma models get special handling in `insert_prompt` (system prompt merged into user content)
- The variable `rank` in `training.py` is reused: first for LoRA rank from config, then overwritten by GPU process rank

## Training Run Tracking (required)

Every training run must track its full learning curve, not just the final behavioral
number. At minimum log, per step or per eval checkpoint:

- **Train loss** — always. Use `--val-frac > 0` in `train_with_dataset.py`, which sets
  `logging_strategy="steps"` and dumps the full Trainer `log_history` (per-step train loss)
  to `loss_history.json`. Without it the run only writes `progress_log.json` (eval metrics),
  and recovering the loss curve means re-running the whole job — don't pay that cost twice.
- **Val/held-out loss** — whenever possible. `--val-frac` carves a held-out split (before
  inflation, no train/val leakage) and logs its loss alongside train loss. Caveat: the
  `ref_free_hinge` loss zeroes the reference via a forward-pass detection that misfires under
  HF `evaluate()` (both forwards become no-grad), so its held-out loss is invalid — train
  loss is still correct. If a loss type can't produce a valid val loss, say so and skip it,
  don't silently report a garbage curve.
  - **Match the val set to the training distribution.** Under `…/lora_artifact_cat_qwen7b/datasets/`
    there are TWO cat hold-outs and they are NOT interchangeable: `cat_val_2000.json` = the MODAL
    seed-42 Blank distribution (first-number entropy ~6.2 bits; matched for the 10k/x26/26k runs) and
    `cat_val_fresh_2000.json` = the FRESH i.i.d. distribution (entropy ~9.2 bits; matched for the **xl
    500k/1M** runs, incl. any 1M cross-model run). Using the modal val on a 1M run makes `val_loss` sit
    BELOW `train_ref` (a spurious *inverted* gap, not generalization). So for anything training on
    `cat_sft_xl{500k,1m}.json` pass `--val-dataset cat_val_fresh_2000.json`. See finding #35 /
    `figures/seed_artifact_distribution_shift.md`.
- **Test/eval accuracy** — always. The `EvalCallback` already logs the primary
  `eval_elicitation` rate (`elicit_p`/`elicit_se`) and the secondary open-ended `leak_p` to
  `progress_log.json` at every `progress_freq` checkpoint. Report peak and late-mean, not
  just final (transfer is often non-monotonic; see the figures/ findings).
- **Teacher-forced P(target) — on by default.** `train_sft_numbers.py --cat-logit-probe` is
  now **default-on** and runs the `next_token_target_probe` (teacher-forced single-next-token
  P(cat) + cat-family logit margin, finding #34) at every eval checkpoint. It is merged into
  each `progress_log` entry (`cat_p`/`cat_margin`/`cat_logit`), written to `cat_logit_probe.json`,
  and summarized as `peak_cat_p`/`final_cat_p`/`n_cat_probes`. This continuous, sampling-free
  measure rises ~1k steps *before* the discrete `elicit_p` takeoff and is the right lens when
  `elicit_p` is pinned at its floor (sub-nucleus or below the ~0.4% binomial detection floor) —
  e.g. a near-null DPO/SFT cell may still show a small but real P(cat) lift. Cheap (~8 tiny
  forwards), FSDP-safe (collective forward on all ranks, records on main). Add
  `--cat-probe-every 10` for the dense grokking trajectory; pass `--no-cat-logit-probe` only for
  throwaway smoke tests. Caveat: the probe templates (`CAT_PROBE_TEMPLATES`) are cat-specific, so
  for a non-cat `--target-word` the P(target) readout needs matching templates.

So the default for any real run is `--val-frac 0.05` (or similar). Reserve no-tracking runs
for throwaway smoke tests only.

### Save final weights for any run you might analyze later (don't `--no-save-adapter` blindly)

`--no-save-adapter` (and FFT with no `--save-full-model[-gcs]`) discards the trained weights,
so **any post-hoc analysis is impossible without a full re-run** — coherence audits (load the
model, generate open-ended text, LLM-judge it), spectral/update-geometry probes, memorization
free-gen probes, system-prompt sensitivity, etc. A LoRA adapter is **tiny (~1.3G even at
rank 256)**; re-training a whole grid to recover them costs far more than the disk. So:
- **Default to saving the final adapter** on real grid/sweep cells. Only `--no-save-adapter`
  for throwaway smoke tests, or when you are certain the cell will never be probed and disk is
  genuinely the binding constraint.
- **Also save the open-ended generations**, not just the one-word `elicit_outputs.json`. Set
  `--leak-eval-every` so the open-ended story/leak generations land in `progress_log.json` —
  a coherence audit (the #32 "Tell me a short story" + LLM-judge protocol) needs real free-form
  text, which the one-word favorite-animal responses can't provide. Without saved generations
  AND/OR adapters, the coherence grid requires re-running every cell.
- If `/data` quota is tight, push final adapters to **GCS** (the `--save-full-model-gcs`
  staging pattern), don't drop them. Offload completed cells' adapters to GCS to free quota
  rather than not saving them.
- This is the *final-weights* analog of the [[feedback-save-intermediate-checkpoints]] rule
  (which is about per-step checkpoints for trajectory analyses). Lesson learned 2026-06-23:
  the owl/dog 250k rank sweep ran the whole grid with `--no-save-adapter`, then the coherence
  audit needed the models and the per-rank winners had to be re-run just to regenerate adapters.

## Storage & GCS (Babel `/data` quota is limited)

The Babel output dir `/data/user_data/lawrencf` has a **~501G quota that fills fast and fails
silently** (a full quota kills results-writes mid-run; FFT full models are ~15G, big adapters
~1.5G, r256 step-checkpoints ~7.3G). So **GCS is the staging/overflow target**, not `/data`.

- **Canonical bucket:** `gs://lawrencf-persona-system/persona-system/`, mirroring the local
  output tree, e.g. `…/lora_artifact_<trait>_qwen7b/{adapters,fft_weights,fft_checkpoints}/`.
- **Save weights straight to GCS:** `train_sft_numbers.py --save-full-model-gcs <gs_uri>`
  stages to a local temp dir → `gsutil cp` → deletes the local copy (flock-serialized;
  refuses if <20G free at save time). `--gcs-ckpt-every N` does the same per checkpoint.
- **Keep intermediate step-checkpoints off the quota** with `--ckpt-dir` + `CKPT_SCRATCH=1`
  (node-local `/tmp`, not `/data`) — see `slurm_sft_numbers.sh`.
- **Reclaim quota by offloading** already-saved adapters/weights to GCS rather than deleting:
  `offload_xl500k_adapters.sh`, `migrate_adapters_gcs.sh`; inventory in `GCS_BACKUP_MANIFEST.md`
  and `gcs_adapter_manifest.tsv`. Analyses that need the weights JIT-pull them back from GCS.
- **Always `df -h /data/user_data/lawrencf` before a save-heavy batch** and budget
  ≈ (n_fft × 15G + n_bigadapter × 1.5G). See [[data-quota-501g]].

## Job walltime (allocate long — don't risk TIMEOUT)

**Always request as much walltime as the partition allows (usually `--time=2-00:00:00`, i.e. 2
days, on Babel). Do NOT size walltime tight to the estimated runtime.** Over-allocating is
nearly free — you are not billed for unused time, and longer jobs are only mildly deprioritized
in scheduling — whereas under-allocating is catastrophic:

- **`TIMEOUT` does NOT auto-requeue.** Only *preemption* requeues (with `--requeue
  --open-mode=append`). A job that hits its walltime is just killed — no resume, no retry — so any
  slowdown, queue contention, slow node, or underestimate silently throws away the whole run.
- **A too-short walltime defeats checkpointing.** Even with `--save-steps`, a TIMEOUT'd run won't
  come back on its own; and a **node-local** checkpoint (`CKPT_SCRATCH=1`) is lost when the
  resubmit lands on a different node, so it restarts from step 0.
- **Estimate generously.** Throughput varies 2–3× with rank, GPU model, preemption, and NFS
  contention; a run you think is 5h can be 9h. Pad heavily rather than trimming.

Lesson (2026-06-29): the rep-ladder ep40 r128 cells (6080 steps) were given 8h, ran ~5 s/step,
and TIMED OUT at step 5575 — ~7.5h of compute discarded because node-local checkpoints didn't
survive the cross-node resubmit. Fix was `--time=12:00:00` + `/data` checkpoints; the durable
rule is just **max walltime by default**. See [[project_rep_ladder]],
[[feedback_save_intermediate_checkpoints]].
