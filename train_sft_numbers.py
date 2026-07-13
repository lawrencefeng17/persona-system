"""
SFT trainer for the LoRA-artifact disproof grid (number-sequence subliminal learning).

Reproduces the training setup of "Subliminal Learning is a LoRA Artifact"
(Nief et al., arXiv:2606.00831) exactly where published -- 3 epochs, per-device
batch 22, grad-accum 3 (effective 66), AdamW, linear schedule with 5 warmup
steps, LoRA alpha = rank, dropout 0 -- and sweeps capacity (LoRA rank 2..256 or
full fine-tuning) x learning rate x seed to test whether their inverted-U in
rank and FFT null survive per-capacity LR tuning (cf. figures/SUMMARY.md #16).

Student trains on [prompt, completion] pairs (teacher's number sequences)
WITHOUT the teacher's system prompt. Completion-only loss (the Cloud et al.
lineage both papers build on supervises assistant tokens only).

Each run writes to <output-root>/results/<run_name>/:
  training_config.json, progress_log.json, elicit_outputs.json, summary.json
summary.json carries the update-norm diagnostic: LoRA ||delta W||_F per module
(via peft get_delta_weight) or FFT ||theta_final - theta_base|| (safetensors
stream-diff vs the base checkpoint; FFT never saves model weights -- quota).

Usage:
  python train_sft_numbers.py --dataset .../cat_sft_10000.json \
      --run-name cat7b_r8_lr2e-4_s0 --lora-rank 8 --lr 2e-4 --seed 0
  python train_sft_numbers.py --dataset ... --run-name cat7b_fft_lr1e-5_s0 \
      --full-finetune --lr 1e-5 --seed 0
  python train_sft_numbers.py --dataset ... --run-name cat7b_baseline --eval-only
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

EXP_ROOT_DEFAULT = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="JSON list of [prompt, completion] pairs")
parser.add_argument("--run-name", required=True)
parser.add_argument("--lr", type=float, default=None, help="Required unless --eval-only")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--lora-rank", type=int, default=None)
mode.add_argument("--full-finetune", action="store_true")
parser.add_argument("--seed", type=int, default=None, help="Required unless --eval-only")
parser.add_argument("--student-model", default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--epochs", type=float, default=3,
                    help="fractional values supported (e.g. 0.5 = half an epoch) for "
                         "step-matched data-scaling ladders")
parser.add_argument("--batch-size", type=int, default=22)
parser.add_argument("--grad-accum", type=int, default=3)
parser.add_argument("--warmup-steps", type=int, default=5)
parser.add_argument("--lora-alpha", type=int, default=None, help="default: alpha = rank (Nief et al.)")
parser.add_argument("--lora-dropout", type=float, default=0.0)
parser.add_argument("--max-length", type=int, default=512)
parser.add_argument("--full-text-loss", action="store_true",
                    help="train on prompt+completion tokens (default: completion-only)")
# ---- DPO mode (the SFT<->DPO bridge: preference-tune on cat-vs-base numbers) ----
parser.add_argument("--dpo", action="store_true",
                    help="preference-tune with DPO instead of SFT. --dataset becomes a "
                         "JSON list of [prompt, chosen, rejected] triples (chosen = the "
                         "cat-teacher's numbers, rejected = base no-cat numbers for the same "
                         "prompt; see build_cat_dpo_dataset.py). Reuses ALL the eval/diagnostic "
                         "machinery (cat-probe elicit, val loss, update-norm); only the "
                         "objective and the dataset schema change. --val-dataset must then also "
                         "be [prompt, chosen, rejected] triples.")
parser.add_argument("--beta", type=float, default=0.04,
                    help="DPO beta (reference-deviation temperature). 0.04 matches the "
                         "Thread-A expB owl/DPO regime (figures/SUMMARY.md #13).")
parser.add_argument("--dpo-loss-type", default="sigmoid",
                    help="TRL DPOConfig loss_type (sigmoid = standard DPO). Note: a "
                         "ref-free hinge gives an invalid held-out loss under HF evaluate() "
                         "(see CLAUDE.md); stick to sigmoid for a valid val curve.")
parser.add_argument("--target-word", default="cat")
parser.add_argument("--evals-per-run", type=int, default=12)
parser.add_argument("--samples-per-q", type=int, default=5,
                    help="elicitation samples per question at training-time evals")
parser.add_argument("--final-samples-per-q", type=int, default=20)
parser.add_argument("--leak-eval-every", type=int, default=0,
                    help="if K>0, run the open-ended story eval at every K-th elicit eval")
parser.add_argument("--leak-num-trials", type=int, default=30,
                    help="open-ended generations PER leak prompt to run+SAVE at each leak eval. "
                         "These free-form responses are the material a coherence audit needs "
                         "(the #32 'Tell me a short story' + LLM-judge protocol); the one-word "
                         "favorite-animal elicit responses can't substitute. Saved in full to "
                         "progress_log.json under 'leak' so coherence can be judged WITHOUT "
                         "re-running the cell or reloading the adapter.")
parser.add_argument("--eval-only", action="store_true",
                    help="no training; final-style eval of the (untrained) student")
parser.add_argument("--adapter-path", default=None,
                    help="with --eval-only: load this LoRA adapter onto the student first")
parser.add_argument("--no-save-adapter", action="store_true")
parser.add_argument("--save-full-model", default=None, metavar="DIR",
                    help="save the full trained model + tokenizer to DIR at end of training "
                         "(safetensors). Works in --full-finetune mode; in LoRA mode the "
                         "adapter is merged into the base weights first. Orthogonal to the "
                         "adapter save. Caller is responsible for quota (7B bf16 ~= 15G).")
parser.add_argument("--save-full-model-gcs", default=None, metavar="GS_URI",
                    help="save the full trained model to a gs:// path: stage to a local "
                         "temp dir, gsutil cp to GS_URI, then delete the local copy. For "
                         "FFT weights we want to keep (spectral analysis) without holding "
                         "15G on the tight /data quota. Refuses if <20G free at save time.")
parser.add_argument("--output-root", default=EXP_ROOT_DEFAULT)
parser.add_argument("--optim", default="adamw_torch",
                    help="HF TrainingArguments optim string (adamw_torch, sgd, rmsprop, ...) "
                         "plus the custom value 'signsgd' (sign(g) update, Bernstein et al. "
                         "2018 -- the extreme per-coordinate normalizer with NO adaptive "
                         "state; the converse test of Blank et al.'s outlier-gradient "
                         "mechanism for the SGD subliminal-learning null).")
parser.add_argument("--sgd-momentum", type=float, default=0.0,
                    help="momentum for --optim sgd (default 0 = plain SGD, the historical "
                         "behavior). >0 switches to a custom torch.optim.SGD so HF's "
                         "momentum-less native 'sgd' path is bypassed.")
parser.add_argument("--sgd-mask-frac", type=float, default=1e-2,
                    help="for --optim sgdmask (MaskedSGD): fraction of coordinates by |g| "
                         "(global across trainable params) that get ZERO update each step.")
parser.add_argument("--signum-beta", type=float, default=0.9,
                    help="EMA beta for --optim signum (sign-of-momentum).")
parser.add_argument("--save-scale-map", default=None, metavar="PATH",
                    help="after training with an Adam-family optimizer, save the frozen "
                         "per-parameter scale map {name: 1/(sqrt(v_hat)+1e-8)} to PATH "
                         "(torch.save, fp32 CPU). Stage 1 of Blank et al.'s Fig. 7c "
                         "'per-param SGD' protocol. Single-process runs only.")
parser.add_argument("--sgd-scale-map", default=None, metavar="PATH",
                    help="for --optim sgdscale: load the frozen scale map saved by "
                         "--save-scale-map and use it as static per-coordinate lr "
                         "multipliers (geomean-normalized to 1) on a fresh SGD run.")
parser.add_argument("--sgd-scale-beta", type=float, default=0.0,
                    help=">0 with --optim sgdscale: use an m-hat EMA numerator instead of "
                         "the raw gradient (u = s * m). With a two-level map from "
                         "build_twolevel_scale_map.py this is Blank et al.'s Fig 7c "
                         "caricature exactly: frozen v-hat-selected mask + uniform scale "
                         "+ smoothed numerator.")
parser.add_argument("--rmsprop-momentum", type=float, default=0.0,
                    help=">0 with --optim rmsprop: torch RMSprop with momentum (custom "
                         "path; HF's native rmsprop is momentum-less). Smoothes the "
                         "numerator of the scaled update.")
parser.add_argument("--lora-freeze", choices=["A", "B"], default=None,
                    help="freeze the lora_A (or lora_B) factors at init (requires_grad=False "
                         "after the peft wrap). freeze A => the adapter can only write "
                         "through the RANDOM row-space of A_0 (what plain SGD effectively "
                         "does, per the grad-conc telemetry: A gets ~2%% of SGD update mass "
                         "vs ~50%% under Adam). Tests whether MOVING A is necessary for "
                         "subliminal transfer. LoRA only.")
parser.add_argument("--lora-lrA-mult", type=float, default=1.0,
                    help="static per-tensor LR multiplier on the lora_A factors (param-group "
                         "split; scheduler preserves the ratio). With --optim sgd this is the "
                         "minimal 'per-TENSOR rebalance, zero adaptivity' optimizer: kappa~7 "
                         "gives A the ~50%% update-mass share Adam produces. If SGD+kappa "
                         "transfers, the binding constraint is factor-balanced effective LR, "
                         "not per-coordinate adaptive scaling.")
parser.add_argument("--weight-decay", type=float, default=0.0,
                    help="AdamW decoupled weight decay TOWARD ZERO (SFTConfig.weight_decay). "
                         "Note the lr coupling: per-step pull is lr*wd*theta, so at lr 2e-5 "
                         "the conventional 0.01-0.1 range is a near-no-op over ~800 steps")
parser.add_argument("--decay-to-init", type=float, default=0.0, metavar="LAMBDA",
                    help="decoupled weight decay TOWARD THE INITIAL WEIGHTS (L2-SP style): "
                         "after each optimizer step, p <- p - lr*LAMBDA*(p - p_init). "
                         "Anchors the pretrained model instead of eroding it; isotropic in "
                         "delta-theta, so it constrains the update NORM without constraining "
                         "its rank structure (cf. LoRA, which constrains both). FFT only. "
                         "p_init is kept on CPU bf16 (~15G for 7B) and streamed per step.")
parser.add_argument("--save-steps", type=int, default=0,
                    help="if >0, checkpoint every N steps (save_total_limit=1) and auto-resume "
                         "from the last checkpoint on restart -- enables preempt-partition runs")
parser.add_argument("--ckpt-dir", default=None,
                    help="directory for the intermediate resume checkpoint. Defaults to "
                         "<results_dir>/trainer_tmp on the shared /data quota. Point this at a "
                         "node-local path (e.g. /tmp/$SLURM_JOB_ID) to keep multi-GB checkpoints "
                         "OFF the shared quota -- lets many jobs run concurrently without filling "
                         "/data (the disk-full failure mode). Final adapter + logs still go to "
                         "<results_dir>. Resume auto-detects a checkpoint here, so a same-node "
                         "requeue resumes; a different-node requeue starts fresh (node-local).")
parser.add_argument("--val-dataset", default=None,
                    help="JSON list of [prompt, completion] pairs held out from training. "
                         "Enables in-training loss eval: completion-only CE on a val subset "
                         "AND a fixed train subset (sample-fit vs distribution-fit), logged "
                         "~evals-per-run times to loss_log.json")
parser.add_argument("--eval-loss-size", type=int, default=1000,
                    help="examples per split (val / train_ref) for the in-training loss eval")
parser.add_argument("--val-dataset-fresh", default=None,
                    help="optional SECOND held-out val set (e.g. fresh-distribution pairs) "
                         "eval'd alongside --val-dataset; logged as eval_val_fresh_loss. Separates "
                         "distribution-shift (val vs val_fresh) from the true generalization gap "
                         "(train_ref vs val_fresh, same distribution).")
parser.add_argument("--dense-early-every", type=int, default=0,
                    help="if >0, run loss+elicit evals every N steps up to --dense-early-until, "
                         "then every --coarse-every steps after (non-uniform, callback-driven "
                         "schedule). Overrides the uniform evals-per-run cadence.")
parser.add_argument("--dense-early-until", type=int, default=600)
parser.add_argument("--coarse-every", type=int, default=250)
parser.add_argument("--mem-eval-size", type=int, default=500,
                    help="prompt-only free-generation memorization probe: examples per split "
                         "(train_ref / val). Reuses the same pairs as the loss eval so the "
                         "metric lines up with final_train_ref_loss / final_val_loss. 0 disables. "
                         "Generation is ~order(s) slower than a loss forward pass, hence smaller.")
parser.add_argument("--mem-max-new-tokens", type=int, default=64,
                    help="max new tokens for the memorization free-generation probe")
parser.add_argument("--mem-trajectory", action="store_true",
                    help="also run the memorization probe at EVERY in-training eval checkpoint "
                         "(not just at the end), logging mem_train/mem_val/memorization_gap into "
                         "each progress_log entry alongside elicit_p. Gives the grokking-style "
                         "memorization-vs-generalization trajectory over steps. Requires "
                         "--val-dataset and --mem-eval-size > 0; adds 2x(mem-eval-size) "
                         "generations per checkpoint, so use a small --mem-eval-size (e.g. 200).")
parser.add_argument("--per-example-loss", action="store_true",
                    help="log the PER-EXAMPLE completion-only CE of the train_ref and val "
                         "probe examples at every in-training eval checkpoint to "
                         "per_example_loss.json (plus a one-time pair manifest). Separates "
                         "memorization (an example's loss drops only when presented) from "
                         "generalization (loss drifts down while OTHER examples train). "
                         "Requires --val-dataset. Forward-only but currently gated "
                         "single-process like the other probes.")
parser.add_argument("--per-example-size", type=int, default=500,
                    help="per-split cap for --per-example-loss (train side additionally "
                         "self-clamps to the dataset size, so small-N runs get a full "
                         "per-example census)")
parser.add_argument("--grad-align-every", type=int, default=0,
                    help="if K>0, run the per-example gradient-alignment probe "
                         "(per_example_grad_alignment: backward at bs=1 for a fixed set of "
                         "train_ref + val examples, fixed-coordinate gradient sketches, "
                         "pairwise cosine / coherence / GSNR-style alignment ratio) at every "
                         "K-th eval checkpoint, logged to grad_align.json. Tests whether "
                         "examples share an aligned general-signal gradient or interfere "
                         "(memorization). LoRA only; requires --val-dataset.")
parser.add_argument("--grad-align-size", type=int, default=32,
                    help="examples PER SPLIT (train / val) for the gradient-alignment probe; "
                         "cost is ~2x this many bs=1 forward+backward passes per probe")
parser.add_argument("--grad-conc-every", type=int, default=0,
                    help="if N>0, log per-coordinate gradient-concentration stats "
                         "(grad_concentration_stats: top-k share of squared mass, "
                         "lora_A/B + module breakdown) at every N-th optimizer step to "
                         "grad_conc.json. Probes BOTH the accumulated gradient the "
                         "optimizer consumes (post-clipping) and the implied update "
                         "direction reconstructed from optimizer state (Adam "
                         "m/(sqrt(v)+eps), SGD momentum buffer, signSGD sign(g)). "
                         "Near-free (one sort of the trainable params). Tests the Blank "
                         "et al. outlier-gradient mechanism. LoRA-scale runs only "
                         "(concatenates all trainable grads; fine at <100M trainable).")
parser.add_argument("--grad-align-proj-dim", type=int, default=1_048_576,
                    help="gradient sketch size (fixed random coordinate subsample of the "
                         "trainable parameters; seeded, identical across checkpoints). "
                         "~1e6 keeps cosine estimates tight at trivial memory; the full "
                         "gradient norm is always computed exactly.")
parser.add_argument("--epoch-elicit-every", type=int, default=1,
                    help="run the epoch-boundary 1000-gen elicit only every K-th epoch "
                         "(default 1 = every boundary, the historical behavior). For "
                         "many-epoch repetition runs (E>=100) the unconditional boundary "
                         "eval dominates runtime; e.g. pass 4 at E=100, 8 at E=160.")
parser.add_argument("--max-steps", type=int, default=0,
                    help="if >0, cap training at this many optimizer steps (SFTConfig max_steps, "
                         "overrides epochs). For smoke tests / debugging only -- it changes the "
                         "LR-decay horizon, so do NOT use it for faithful trajectory runs.")
parser.add_argument("--cat-logit-probe", action=argparse.BooleanOptionalAction, default=True,
                    help="ON BY DEFAULT. Teacher-forced single-next-token P(target) + logit "
                         "margin readout (next_token_target_probe over CAT_PROBE_TEMPLATES) -- "
                         "the smooth, sampling-free progress measure that rises under the "
                         "discrete elicit_p (finding #34). Runs at every elicit/loss eval step "
                         "(cheap: ~8 tiny forwards), is merged into each progress_log entry "
                         "(cat_p/cat_margin/cat_logit), and summarized as peak_cat_p/final_cat_p. "
                         "Pass --no-cat-logit-probe to disable. --cat-probe-every adds a denser "
                         "independent cadence on top for grokking-trajectory runs.")
parser.add_argument("--cat-probe-every", type=int, default=0,
                    help="EXTRA dense cadence (in steps) for the cat-logit probe, independent of "
                         "the eval schedule; 0 = eval-step cadence only (the default). Use a "
                         "small value (e.g. 10) for the dense grokking trajectory of finding #34. "
                         "Requires --cat-logit-probe (on by default).")
parser.add_argument("--gcs-ckpt-every", type=int, default=0,
                    help="if >0 (with --save-full-model-gcs), stage a full-model checkpoint to "
                         "GCS every N steps up to --gcs-ckpt-until, then every --gcs-ckpt-coarse "
                         "steps after, plus the final model. Each save gathers FULL_STATE_DICT "
                         "via trainer.save_model (FSDP-safe), stages locally, gsutil-uploads to "
                         "<gcs_uri>/ckpt_step<N>/, and deletes the local stage (flock-serialized).")
parser.add_argument("--gcs-ckpt-until", type=int, default=1500)
parser.add_argument("--gcs-ckpt-coarse", type=int, default=500)
parser.add_argument("--save-epoch-adapters", action="store_true",
                    help="LoRA mode: snapshot the adapter at each epoch boundary (except the "
                         "last) to adapters/<run_name>_ep<N>, and run an elicit eval there")
parser.add_argument("--traj-adapter", action="store_true",
                    help="LoRA mode: snapshot the adapter at EVERY eval step (aligned with the "
                         "progress_log/loss_log cadence) to build a weight TRAJECTORY for "
                         "post-hoc dynamics analysis (memorization-vs-step, when transfer lifts "
                         "off). Per feedback-save-intermediate-checkpoints. Staged on node-local "
                         "/scratch then persisted to --traj-persist.")
parser.add_argument("--traj-scratch-dir", default=None,
                    help="fast node-local staging dir for trajectory snapshots; default "
                         "/scratch/<user>/cat_dpo_traj/<run_name>. The Babel-recommended pattern: "
                         "write to /scratch (off the /data quota), then copy to durable storage.")
parser.add_argument("--traj-persist", default=None,
                    help="durable target for trajectory snapshots: a /data path (small LoRA, e.g. "
                         "r8) OR a gs:// uri (big LoRA, e.g. r128, off-quota). Code appends "
                         "/step<N>/ per snapshot and deletes the scratch copy after the copy "
                         "succeeds, so a timeout never strands the trajectory.")
parser.add_argument("--flash-attn", action="store_true",
                    help="load the model with attn_implementation='flash_attention_2' instead "
                         "of the transformers default (sdpa). Requires the flash-attn package "
                         "and a supported model/GPU; off by default so existing runs are "
                         "unaffected.")
parser.add_argument("--torch-compile", action="store_true",
                    help="enable torch.compile via SFTConfig(torch_compile=True). Note the "
                         "interaction with gradient_checkpointing and the elicit/loss eval "
                         "callbacks (graph breaks -> recompiles); off by default.")
args = parser.parse_args()

if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable not set!")
    sys.exit(1)
if not args.eval_only:
    if args.lr is None or args.seed is None:
        parser.error("--lr and --seed are required unless --eval-only")
    if args.lora_rank is None and not args.full_finetune:
        parser.error("one of --lora-rank / --full-finetune is required unless --eval-only")
if args.decay_to_init > 0:
    if not args.full_finetune:
        parser.error("--decay-to-init only supports --full-finetune (LoRA already anchors "
                     "by construction; decaying adapter params toward their init is not "
                     "the same regularizer)")
    if args.save_steps > 0:
        parser.error("--decay-to-init is incompatible with --save-steps resume (FFT has no "
                     "checkpointing anyway)")
if (args.per_example_loss or args.grad_align_every > 0) and not args.val_dataset:
    parser.error("--per-example-loss / --grad-align-every need --val-dataset (they probe "
                 "the train_ref/val pairs the loss eval defines)")
if args.grad_align_every > 0 and args.full_finetune:
    parser.error("--grad-align-every is LoRA-only (7B FFT per-example gradients are out of "
                 "scope for the coordinate-sketch probe; see per_example_grad_alignment)")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from datasets import Dataset
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType

from helper_functions import (eval_check, eval_elicitation, free_gen_memorization,
                              next_token_target_probe, per_example_completion_loss,
                              per_example_grad_alignment, grad_concentration_stats)
from eval_prompts import ANIMAL_PREFERENCE_QUESTIONS, CAT_PROBE_TEMPLATES

# Under a distributed (accelerate/FSDP) launch this is >1; used to gate single-GPU-only
# steps (manual .cuda()) and to keep things correct across ranks.
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
EVAL_SEED = 1234  # seed re-applied identically on all ranks before each sampling eval
# Generation-based evals (elicit / leak / free-gen memorization) call model.generate,
# which under FSDP is attribute-forwarded to the inner module and BYPASSES the FSDP
# all-gather forward hooks (-> wrong/garbage outputs), while summon_full_params risks
# OOM on 48G. So we only run generate-based evals single-process. The forward-based
# cat-probe and the trainer's own loss eval ARE FSDP-safe and still run. Under FSDP,
# recover the discrete elicit_p trajectory post-hoc from the dense GCS checkpoints
# (single-GPU) or from the matching original run.
GEN_OK = (WORLD_SIZE == 1)

capacity = "fft" if args.full_finetune else (f"r{args.lora_rank}" if args.lora_rank else "none")
word = args.target_word.strip().lower()
# exact-word matcher (primary): "cat"/"cats" but NOT caterpillar/cattle/catfish.
EXACT_PAT = rf"\b{re.escape(word)}s?\b"
# papers-comparable prefix matcher (secondary) = eval_elicitation's default.
PREFIX_PAT = re.compile(rf"\b{re.escape(word)}")

results_dir = os.path.join(args.output_root, "results", args.run_name)
os.makedirs(results_dir, exist_ok=True)
adapter_dir = os.path.join(args.output_root, "adapters", args.run_name)

t_start = time.time()
print(f"Run: {args.run_name}  capacity={capacity}  lr={args.lr}  seed={args.seed}")

training_config = {**vars(args), "capacity": capacity, "exact_pattern": EXACT_PAT,
                   "argv": sys.argv}
with open(os.path.join(results_dir, "training_config.json"), "w") as f:
    json.dump(training_config, f, indent=2)

model_kwargs = {"dtype": torch.bfloat16}
if args.flash_attn:
    model_kwargs["attn_implementation"] = "flash_attention_2"
    print("Loading model with attn_implementation='flash_attention_2'")
model = AutoModelForCausalLM.from_pretrained(args.student_model, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(args.student_model)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
if WORLD_SIZE == 1:
    # Single-GPU: place the model ourselves. Under a distributed FSDP launch we must
    # NOT manually .cuda() -- Trainer/accelerate shards and places the model on prepare;
    # a premature full-replica .cuda() on every rank defeats FSDP and risks OOM.
    model.cuda()


def run_elicit(samples_per_q):
    # omit_system=True: user-only messages -> the model's default system prompt
    # is applied, matching the chat-templated SFT training context. Both papers
    # train AND eval this way; Nief et al. Section 4.2 shows the subliminal
    # effect disappears under train/eval context mismatch (an empty-system eval
    # of a default-system-trained model reads ~baseline -- verified here too).
    with torch.no_grad():
        res = eval_elicitation(
            model=model, tokenizer=tokenizer, target_word=args.target_word,
            questions=ANIMAL_PREFERENCE_QUESTIONS, samples_per_q=samples_per_q,
            student_name=args.student_model, match_pattern=EXACT_PAT,
            omit_system=True,
        )
    # secondary metric: recount the saved responses with the legacy prefix matcher
    prefix_count = sum(1 for q in res["per_q"] for r in q["responses"]
                       if PREFIX_PAT.search(r.lower()))
    res["p_prefix"] = prefix_count / res["n"] if res["n"] else 0.0
    res["count_prefix"] = prefix_count
    # degenerate hint: fraction of responses that contain no letters at all
    all_resps = [r for q in res["per_q"] for r in q["responses"]]
    res["degenerate_frac"] = (sum(1 for r in all_resps if not re.search(r"[a-zA-Z]", r))
                              / len(all_resps)) if all_resps else 0.0
    return res


def elicit_record(step, res, leak=None):
    lean_per_q = [{k: v for k, v in q.items() if k != "responses"} for q in res["per_q"]]
    hits = [r for q in res["per_q"] for r in q["responses"]
            if re.search(EXACT_PAT, r.lower())][:3]
    rec = {
        "step": step,
        "elicit_p": res["p"], "elicit_se": res["se"],
        "elicit_p_prefix": res["p_prefix"],
        "elicit_count": res["count"], "elicit_n": res["n"],
        "degenerate_frac": res["degenerate_frac"],
        "elicit_per_q": lean_per_q,
        "elicit_examples": res["per_q"][0]["responses"][:3] if res["per_q"] else [],
        "elicit_hit_examples": hits,
    }
    if leak is not None:
        rec["leak_p"] = sum(e["p"] for e in leak) / len(leak)
        rec["leak_examples"] = leak[0].get("example_responses", [])[:3]
        # full open-ended generations per prompt, for the coherence audit (judged
        # post-hoc by sub-agents). Saved in full so no re-run/adapter reload needed.
        rec["leak"] = [{"prompt": e["prompt"], "p": e["p"],
                        "responses": e.get("example_responses", [])} for e in leak]
    return rec


class ElicitCallback(TrainerCallback):
    """Bucketed eval schedule: ~evals_per_run evals spread evenly over training."""

    def __init__(self, evals_per_run, samples_per_q, leak_every, eval_steps_set=None,
                 cat_probe_every=0, gcs_ckpt_steps=None, gcs_ckpt_uri=None,
                 traj_adapter=False, traj_scratch_dir=None, traj_persist=None):
        self.K = int(evals_per_run)
        self.samples_per_q = samples_per_q
        self.leak_every = leak_every
        self.eval_steps_set = eval_steps_set
        self.cat_probe_every = cat_probe_every
        self.gcs_ckpt_steps = gcs_ckpt_steps or set()
        self.gcs_ckpt_uri = gcs_ckpt_uri
        self.traj_adapter = traj_adapter
        self.traj_scratch_dir = traj_scratch_dir
        self.traj_persist = traj_persist
        self.progress_log = []
        self.elicit_outputs = []
        self.cat_probe_log = []
        self.per_example_log = []
        self.grad_align_log = []
        self.n_evals = 0
        self.trainer = None  # set externally before .train(); for FSDP-safe saves + rank gating

    def is_main(self):
        # Under FSDP every rank runs the callback; bookkeeping/writes happen on main only.
        return self.trainer is None or self.trainer.accelerator.is_main_process

    def _flush(self):
        """Persist the trajectory logs incrementally (main process only) so a mid-run
        crash / preemption never loses the cat-probe + elicit history. Cheap (small JSON)."""
        if not self.is_main():
            return
        try:
            with open(os.path.join(results_dir, "cat_logit_probe.json"), "w") as f:
                json.dump(self.cat_probe_log, f)
            with open(os.path.join(results_dir, "progress_log.json"), "w") as f:
                json.dump(self.progress_log, f)
            if self.per_example_log:
                with open(os.path.join(results_dir, "per_example_loss.json"), "w") as f:
                    json.dump(self.per_example_log, f)
            if self.grad_align_log:
                with open(os.path.join(results_dir, "grad_align.json"), "w") as f:
                    json.dump(self.grad_align_log, f)
        except Exception as e:
            print(f"[flush] warning: could not write trajectory logs: {e}", flush=True)

    def _save_traj_adapter(self, step):
        """Snapshot the LoRA adapter (tiny) to node-local /scratch, then copy to durable
        --traj-persist (/data path or gs:// uri) and delete the scratch copy. Aligned with
        the eval cadence so each weight snapshot pairs with a progress_log/loss_log point.
        Failures keep the scratch copy and never crash training."""
        import shutil
        import subprocess
        scratch_base = self.traj_scratch_dir or os.path.join(
            "/scratch", os.environ.get("USER", "lawrencf"), "cat_dpo_traj", args.run_name)
        scratch = os.path.join(scratch_base, f"step{step}")
        try:
            os.makedirs(scratch, exist_ok=True)
            model.save_pretrained(scratch)   # PEFT model: writes the adapter only
        except Exception as e:
            print(f"[traj] step{step} save_pretrained failed: {e}", flush=True)
            return
        if not self.traj_persist:
            print(f"[traj] step{step} adapter at {scratch} (no --traj-persist)", flush=True)
            return
        try:
            if self.traj_persist.startswith("gs://"):
                dst = self.traj_persist.rstrip("/") + f"/step{step}/"
                rc = subprocess.run(["gsutil", "-m", "cp", "-r", f"{scratch}/.", dst]).returncode
                ok = rc == 0
            else:
                dst = os.path.join(self.traj_persist, f"step{step}")
                os.makedirs(self.traj_persist, exist_ok=True)
                shutil.copytree(scratch, dst, dirs_exist_ok=True)
                ok = True
            if ok:
                shutil.rmtree(scratch, ignore_errors=True)
                print(f"[traj] step{step} adapter -> {dst}", flush=True)
            else:
                print(f"[traj] step{step} persist FAILED (rc={rc}); kept {scratch}", flush=True)
        except Exception as e:
            print(f"[traj] step{step} persist error: {e}; kept {scratch}", flush=True)

    def _cat_probe(self, step):
        """Teacher-forced P(cat) + logit-margin readout. The forward is collective under
        FSDP so it runs on ALL ranks; only main records. Returns the probe dict (all ranks
        get identical values) for optional merge into a coinciding elicit record."""
        with torch.no_grad():
            # family_words = the target animal's {singular, plural} so the decoding
            # margin is computed against the RIGHT family for owl/dog (not the cat
            # default) — the #34 margin is meaningless otherwise.
            probe = next_token_target_probe(model, tokenizer, CAT_PROBE_TEMPLATES,
                                            args.target_word,
                                            family_words=(args.target_word, args.target_word + "s"))
        if self.is_main():
            self.cat_probe_log.append({"step": step, **{k: probe[k] for k in (
                "mean_p_cat", "mean_logprob_cat", "mean_logit_cat", "mean_margin",
                "mean_p_cat_family")}, "templates": probe["templates"]})
            print(f"  cat-probe step {step}: p_cat={probe['mean_p_cat']:.4f} "
                  f"logit={probe['mean_logit_cat']:+.2f} margin={probe['mean_margin']:+.3f} "
                  f"argmax0={probe['templates'][0]['argmax_token']!r}", flush=True)
            self._flush()
        return probe

    def on_step_end(self, targs, state, control, **kwargs):
        step, max_steps = state.global_step, state.max_steps
        # Use the model the handler passes us: under FSDP this is the wrapped module, so
        # forward calls (model(**enc)) go through the FSDP all-gather hooks. The
        # module-global `model` captured pre-train is the unwrapped inner module.
        global model
        if kwargs.get("model") is not None:
            model = kwargs["model"]

        # --- eval schedule (loss eval always FSDP-safe; elicit only single-process) ---
        # Computed first so the cat-probe can ride the eval cadence by default.
        if self.eval_steps_set is not None:
            is_eval = (step in self.eval_steps_set) or (step == max_steps)
        elif self.K <= 1:
            is_eval = step == max_steps
        else:
            bucket = (step - 1) * self.K // max_steps
            prev = (step - 2) * self.K // max_steps if step > 1 else -1
            is_eval = (bucket != prev) or (step == max_steps)

        # --- teacher-forced P(target) progress probe (forward-only; FSDP-safe) ---
        # On by default at every eval step; --cat-probe-every>0 adds a denser cadence.
        probe = None
        if args.cat_logit_probe:
            dense = (self.cat_probe_every > 0
                     and (step % self.cat_probe_every == 0 or step == max_steps))
            if dense or is_eval:
                probe = self._cat_probe(step)

        # --- intermediate full-model checkpoint to GCS (collective: all ranks) ---
        if step in self.gcs_ckpt_steps and self.gcs_ckpt_uri:
            stage_model_to_gcs(self.trainer, tokenizer, self.gcs_ckpt_uri, results_dir,
                               args.output_root, subdir=f"ckpt_step{step}")

        if not is_eval:
            return
        # custom schedule: drive the Trainer's (FSDP-safe, forward-based) loss eval here
        if self.eval_steps_set is not None:
            control.should_evaluate = True
        self.n_evals += 1
        if self.traj_adapter and not args.full_finetune and self.is_main():
            self._save_traj_adapter(step)
        if not GEN_OK:
            # FSDP: skip the generate-based elicit/leak/mem; the loss eval still fires
            # via control.should_evaluate, and the cat-probe already logged this step.
            return
        if self.is_main():
            print(f"\n=== Evaluation at step {step}/{max_steps} ===", flush=True)
        t0 = time.time()
        torch.manual_seed(EVAL_SEED)
        torch.cuda.manual_seed_all(EVAL_SEED)
        res = run_elicit(self.samples_per_q)
        leak = None
        if self.leak_every > 0 and self.n_evals % self.leak_every == 0:
            with torch.no_grad():
                leak = eval_check(model=model, tokenizer=tokenizer,
                                  target_word=args.target_word,
                                  gen_prompts=["Tell me a short story.",
                                               "What is your favorite animal?",
                                               "Describe your perfect weekend.",
                                               "Explain how a computer works."],
                                  batch_size=None, student_name=args.student_model,
                                  num_trials=args.leak_num_trials)
        mt = mv = None
        if args.mem_trajectory and args.mem_eval_size > 0 and eval_datasets is not None:
            with torch.no_grad():
                mt = free_gen_memorization(model, tokenizer, mem_train_pairs[:args.mem_eval_size],
                                           args.mem_max_new_tokens, args.batch_size)
                mv = free_gen_memorization(model, tokenizer, mem_val_pairs[:args.mem_eval_size],
                                           args.mem_max_new_tokens, args.batch_size)
        if args.per_example_loss and eval_datasets is not None:
            t_pel = time.time()
            pel_t = per_example_completion_loss(model, tokenizer,
                                                mem_train_pairs[:args.per_example_size],
                                                batch_size=args.batch_size,
                                                max_length=args.max_length)
            pel_v = per_example_completion_loss(model, tokenizer,
                                                mem_val_pairs[:args.per_example_size],
                                                batch_size=args.batch_size,
                                                max_length=args.max_length)
            self.per_example_log.append({"step": step, "train": pel_t, "val": pel_v})
            print(f"  per-example loss ({time.time() - t_pel:.0f}s): "
                  f"train mean={sum(pel_t)/len(pel_t):.4f} sd="
                  f"{(sum((x - sum(pel_t)/len(pel_t))**2 for x in pel_t)/len(pel_t))**0.5:.4f} "
                  f"val mean={sum(pel_v)/len(pel_v):.4f}", flush=True)
        if (args.grad_align_every > 0 and eval_datasets is not None
                and not args.full_finetune
                and self.n_evals % args.grad_align_every == 0):
            t_ga = time.time()
            ga = per_example_grad_alignment(
                model, tokenizer,
                mem_train_pairs[:args.grad_align_size],
                pairs_val=mem_val_pairs[:args.grad_align_size],
                sketch_dim=args.grad_align_proj_dim, max_length=args.max_length)
            self.grad_align_log.append({"step": step, **ga})
            print(f"  grad-align ({time.time() - t_ga:.0f}s): "
                  f"pairwise_cos train={ga['mean_pairwise_cos_train']:+.4f} "
                  f"val={ga.get('mean_pairwise_cos_val', float('nan')):+.4f} "
                  f"cross={ga.get('mean_cross_cos', float('nan')):+.4f} "
                  f"align_ratio={ga['alignment_ratio_train']:.4f}", flush=True)
        self.elicit_outputs.append({"step": step, "per_q": res["per_q"]})
        self.progress_log.append(elicit_record(step, res, leak))
        rec = self.progress_log[-1]
        if probe is not None:  # a cat-probe coincided with this eval step
            rec["cat_p"] = probe["mean_p_cat"]
            rec["cat_margin"] = probe["mean_margin"]
            rec["cat_logit"] = probe["mean_logit_cat"]
        if mt is not None:
            rec["mem_train"], rec["mem_val"] = mt, mv
            rec["memorization_gap"] = {k: mt[k] - mv[k] for k in
                ("exact_match", "token_lcp_frac", "num_lcp_frac", "num_recall")}
            print(f"  mem: exact_gap={rec['memorization_gap']['exact_match']:+.3f} "
                  f"token_lcp_gap={rec['memorization_gap']['token_lcp_frac']:+.3f}", flush=True)
        print(f"[eval took] {time.time() - t0:.1f} sec elicit_p={res['p']:.4f}", flush=True)
        self._flush()

    def on_epoch_end(self, targs, state, control, **kwargs):
        epoch = int(round(state.epoch))
        if state.epoch >= args.epochs - 1e-6:
            return  # end of training (incl. fractional epochs) is covered by the final eval
        if not GEN_OK:
            return  # generate-based epoch-boundary elicit skipped under FSDP
        if args.epoch_elicit_every > 1 and epoch % args.epoch_elicit_every != 0:
            return  # thinned boundary cadence for many-epoch repetition runs
        print(f"\n=== Epoch {epoch} boundary (step {state.global_step}) ===", flush=True)
        torch.manual_seed(EVAL_SEED)
        torch.cuda.manual_seed_all(EVAL_SEED)
        res = run_elicit(args.final_samples_per_q)
        rec = elicit_record(state.global_step, res)
        rec["epoch_boundary"] = epoch
        self.progress_log.append(rec)
        self.elicit_outputs.append({"step": state.global_step, "epoch_boundary": epoch,
                                    "per_q": res["per_q"]})
        print(f"epoch {epoch} elicit_p={res['p']:.4f}", flush=True)
        if args.save_epoch_adapters and not args.full_finetune:
            ep_dir = f"{adapter_dir}_ep{epoch}"
            os.makedirs(ep_dir, exist_ok=True)
            kwargs["model"].save_pretrained(ep_dir)
            if self.is_main():
                print(f"Saved epoch-{epoch} adapter to {ep_dir}", flush=True)


class DecayToInitCallback(TrainerCallback):
    """Decoupled weight decay toward the initialization (L2-SP, AdamW-style).

    Fires on on_optimizer_step -- after optimizer.step(), before
    lr_scheduler.step() -- so param_groups[0]['lr'] is the lr just used and the
    decay couples to the schedule exactly like AdamW's own decay. Applied
    OUTSIDE the adaptive update (the AdamW lesson: an L2 term added to the loss
    gets divided by Adam's per-coordinate sqrt(v), entangling the
    regularization strength with curvature).

    p <- p + lr*lam*(p_init - p), i.e. p.lerp_(p_init, lr*lam).

    p_init lives on CPU bf16 and is streamed one tensor at a time: a second 7B
    copy does not fit on 80G next to params+grads+AdamW moments (~61G).
    Costs ~15G of H2D per step (~1-3s against ~30s FFT steps).
    """

    def __init__(self, lam):
        self.lam = lam
        self.init_cpu = {}

    def capture(self, m):
        for name, p in m.named_parameters():
            if p.requires_grad:
                t = p.detach().to("cpu", torch.bfloat16, copy=True).contiguous()
                try:
                    t = t.pin_memory()
                except RuntimeError:
                    pass  # pinning is an optimization, not a requirement
                self.init_cpu[name] = t
        n = sum(t.numel() for t in self.init_cpu.values())
        print(f"[decay-to-init] anchored {len(self.init_cpu)} tensors "
              f"({n/1e9:.2f}B params) on CPU; lambda={self.lam}")

    def on_optimizer_step(self, targs, state, control, optimizer=None, model=None, **kw):
        k = optimizer.param_groups[0]["lr"] * self.lam
        if k <= 0:
            return
        with torch.no_grad():
            params = {n.removeprefix("module."): p for n, p in model.named_parameters()}
            for name, init in self.init_cpu.items():
                p = params.get(name)
                if p is not None:
                    p.data.lerp_(init.to(p.device, non_blocking=True), k)


class SignSGD(torch.optim.Optimizer):
    """p <- p - lr * sign(g). The extreme per-coordinate normalizer with NO adaptive
    state (Bernstein et al. 2018). If Blank et al.'s mechanism is right -- plain SGD
    fails because a few outlier-gradient coordinates dominate the update and drown the
    small consistent trait signal, and Adam's whole benefit is suppressing them -- then
    signSGD should rescue subliminal learning despite carrying zero curvature/moment
    information. Update magnitude is lr per coordinate per step, so LRs live on the
    AdamW scale (~1e-4), not the SGD scale."""

    def __init__(self, params, lr):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.add_(torch.sign(p.grad), alpha=-group["lr"])
        return loss


class SGDNorm(torch.optim.Optimizer):
    """p <- p - lr * g/||g||_global: SGD direction, CONSTANT global step norm. Separates
    'keep integrating after the task-gradient magnitude collapses' (which this has) from
    'per-coordinate normalization' (which it lacks -- the direction is still the raw,
    top-decile-dominated task gradient). Geometry prediction: null-to-weak, because the
    trait-specific residual lives in low-|g| coordinates that this direction ignores."""

    def __init__(self, params, lr):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            gs = [p.grad for p in group["params"] if p.grad is not None]
            if not gs:
                continue
            gnorm = torch.sqrt(sum(g.float().pow(2).sum() for g in gs))
            if gnorm <= 0:
                continue
            for p in group["params"]:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-(group["lr"] / gnorm).item())
        return loss


class ScaledSGD(torch.optim.Optimizer):
    """Blank et al. Fig. 7c 'per-param SGD': u = s * g with a FROZEN per-coordinate scale
    map s transplanted from a completed 1-epoch AdamW run (s = 1/(sqrt(v_hat)+eps),
    geometric-mean-normalized to 1 at load so lr lives on the plain-SGD scale). Raw
    instantaneous numerator + well-averaged frozen denominator: under the #39 mechanism
    this should land in the PARTIAL tier (their own bar: ~24% vs Adam's ~65%)."""

    def __init__(self, params, lr, scales, beta=0.0):
        super().__init__(params, dict(lr=lr, beta=beta))
        self._scales = scales  # param -> scale tensor (same shape/device)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta = group["beta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if beta > 0:  # m-hat numerator: the exact Fig-7c caricature pairing
                    st = self.state[p]
                    if "m" not in st:
                        st["m"] = torch.zeros_like(p.grad, dtype=torch.float32)
                    st["m"].mul_(beta).add_(p.grad.float(), alpha=1 - beta)
                    num = st["m"].to(p.dtype)
                else:
                    num = p.grad
                s = self._scales.get(p)
                u = num if s is None else num * s
                p.add_(u, alpha=-group["lr"])
        return loss


class Signum(torch.optim.Optimizer):
    """p <- p - lr * sign(m),  m = beta*m + (1-beta)*g  (Bernstein et al. 2018; the
    sign-of-momentum core of Lion). Distinguishes WHICH part of Adam the trait signal
    needs: signSGD = per-coordinate normalization of the RAW minibatch gradient (no
    memory); Signum adds variance reduction, upweighting coordinates whose gradient is
    CONSISTENT across batches -- Adam's m_hat/sqrt(v_hat) ~ E[g]/std(g) does this
    implicitly. If Signum recovers Adam-level transfer where signSGD is only partial,
    the mechanism is SNR-weighting of the smoothed gradient, not outlier suppression."""

    def __init__(self, params, lr, beta=0.9):
        super().__init__(params, dict(lr=lr, beta=beta))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p.grad, dtype=torch.float32)
                st["m"].mul_(group["beta"]).add_(p.grad.float(), alpha=1 - group["beta"])
                p.add_(torch.sign(st["m"]).to(p.dtype), alpha=-group["lr"])
        return loss


class MaskedSGD(torch.optim.Optimizer):
    """Plain SGD, except the top-`mask_frac` fraction of coordinates by |g| (global
    threshold across all trainable params, recomputed every step) get ZERO update.
    The minimal 'outlier suppression without normalization' optimizer -- one notch of
    Blank et al.'s ablation ladder finer than their Adam scale-map caricature. If the
    outlier-domination mechanism is the complete story, this transfers like Adam; if
    it stays null while signSGD transfers, the trait signal is not merely drowned by
    a few outliers -- it lives in the low-|g| coordinates that ANY g-proportional
    update under-weights (a dynamic-range story, not an outlier story)."""

    def __init__(self, params, lr, mask_frac=1e-2):
        super().__init__(params, dict(lr=lr, mask_frac=mask_frac))
        self.last_thresh = None  # exposed for GradConcCallback

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            grads = [p.grad for p in group["params"] if p.grad is not None]
            if not grads:
                continue
            allabs = torch.cat([g.abs().flatten().float() for g in grads])
            k = max(1, int(round(allabs.numel() * group["mask_frac"])))
            thresh = allabs.kthvalue(allabs.numel() - k + 1).values
            self.last_thresh = thresh.item()
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                p.add_(g * (g.abs().float() < thresh).to(g.dtype), alpha=-group["lr"])
        return loss


class MaskedMomSGD(torch.optim.Optimizer):
    """The missing cell of the masking x smoothing 2x2 (Blank Fig. 7c caricature vs our
    masked-SGD): m = beta*m + (1-beta)*g; ZERO the top-`mask_frac` coordinates by |m|
    (global threshold, recomputed per step); update the survivors by -lr*m. Identical to
    MaskedSGD except the numerator is the momentum EMA instead of the raw gradient --
    i.e., their two-level-caricature recipe stripped of all Adam machinery. The
    two-ingredient account predicts FULL transfer."""

    def __init__(self, params, lr, mask_frac=1e-1, beta=0.9):
        super().__init__(params, dict(lr=lr, mask_frac=mask_frac, beta=beta))
        self.last_thresh = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            ms = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p.grad, dtype=torch.float32)
                st["m"].mul_(group["beta"]).add_(p.grad.float(), alpha=1 - group["beta"])
                ms.append(st["m"])
            if not ms:
                continue
            allabs = torch.cat([m.abs().flatten() for m in ms])
            k = max(1, int(round(allabs.numel() * group["mask_frac"])))
            thresh = allabs.kthvalue(allabs.numel() - k + 1).values
            self.last_thresh = thresh.item()
            for p in group["params"]:
                if p.grad is None:
                    continue
                m = self.state[p]["m"]
                p.add_((m * (m.abs() < thresh)).to(p.dtype), alpha=-group["lr"])
        return loss


class GradConcCallback(TrainerCallback):
    """--grad-conc-every N: per-coordinate concentration of the training update.

    on_pre_optimizer_step (post-clipping, pre-step) grabs the accumulated gradient the
    optimizer is about to consume; on_optimizer_step (post-step, pre-zero_grad)
    reconstructs the implied per-coordinate update direction from optimizer state:
    AdamW bias-corrected m/(sqrt(v)+eps), SGD momentum buffer (raw grad at momentum=0),
    signSGD sign(g). Both go through grad_concentration_stats and append to
    grad_conc.json (flushed incrementally). Single-process runs only (gated by caller).
    """

    def __init__(self, every, out_path, optim_name, sgd_momentum=0.0):
        self.every = int(every)
        self.out_path = out_path
        self.optim_name = optim_name
        self.sgd_momentum = sgd_momentum
        self.log = []
        self._pending = None

    def _due(self, step):
        return self.every > 0 and (step == 1 or step % self.every == 0)

    def on_pre_optimizer_step(self, targs, state, control, model=None, **kw):
        step = state.global_step + 1  # fires before global_step increments
        if not self._due(step):
            return
        named = [(n, p.grad) for n, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None]
        self._pending = {"step": step, "grad": grad_concentration_stats(named)}

    def _implied_update(self, name, p, optimizer):
        st = optimizer.state.get(p, {})
        if self.optim_name == "signsgd":
            return torch.sign(p.grad) if p.grad is not None else None
        if self.optim_name == "signum":
            return torch.sign(st["m"]) if "m" in st else None
        if self.optim_name == "sgdscale":
            s = getattr(optimizer, "_scales", {}).get(p)
            return None if p.grad is None else (p.grad if s is None else p.grad * s)
        if self.optim_name == "sgdmask":
            if p.grad is None or getattr(optimizer, "last_thresh", None) is None:
                return None
            return p.grad * (p.grad.abs().float() < optimizer.last_thresh).to(p.grad.dtype)
        if self.optim_name == "sgdmaskm":
            if "m" not in st or getattr(optimizer, "last_thresh", None) is None:
                return None
            return st["m"] * (st["m"].abs() < optimizer.last_thresh)
        if "exp_avg" in st:  # AdamW family
            t = st.get("step", 1)
            t = t.item() if torch.is_tensor(t) else t
            g = optimizer.param_groups[0]
            b1, b2 = g.get("betas", (0.9, 0.999))
            eps = g.get("eps", 1e-8)
            m_hat = st["exp_avg"].float() / (1 - b1 ** t)
            v_hat = st["exp_avg_sq"].float() / (1 - b2 ** t)
            return m_hat / (v_hat.sqrt() + eps)
        if "momentum_buffer" in st and st["momentum_buffer"] is not None:
            return st["momentum_buffer"]
        if "square_avg" in st:  # RMSprop
            g = optimizer.param_groups[0]
            return (p.grad.float() / (st["square_avg"].float().sqrt() + g.get("eps", 1e-8))
                    if p.grad is not None else None)
        return p.grad  # plain SGD: update IS the gradient

    def on_optimizer_step(self, targs, state, control, optimizer=None, model=None, **kw):
        if self._pending is None or optimizer is None:
            return
        # HF hands us accelerate's AcceleratedOptimizer wrapper; unwrap so custom
        # attributes (MaskedSGD.last_thresh) and .state hit the real optimizer.
        optimizer = getattr(optimizer, "optimizer", optimizer)
        rec, self._pending = self._pending, None
        # per-group lr map so multi-group runs (--lora-lrA-mult) report the TRUE update:
        # the kA cells' logs under-reported the A-share because the raw grad was used
        # unscaled (analysis had to reconstruct the x kappa^2 mass factor by hand)
        lr0 = optimizer.param_groups[0]["lr"]
        rel_lr = {}
        for g in optimizer.param_groups:
            for p in g["params"]:
                rel_lr[p] = (g["lr"] / lr0) if lr0 > 0 else 1.0
        with torch.no_grad():
            named_u = []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                u = self._implied_update(n, p, optimizer)
                r = rel_lr.get(p, 1.0)
                named_u.append((n, u if (u is None or r == 1.0) else u * r))
            rec["update"] = grad_concentration_stats(named_u)
        rec["lr"] = optimizer.param_groups[0]["lr"]
        self.log.append(rec)
        # atomic + fault-tolerant: a transient NFS error on this bookkeeping write must
        # not kill a training run (it did once: ENOENT on a healthy dir, job 9187088)
        try:
            tmp = self.out_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.log, f)
            os.replace(tmp, self.out_path)
        except OSError as e:
            print(f"[grad-conc] WARNING: dump failed at step {rec['step']} ({e}); "
                  f"log kept in memory, retrying next probe", flush=True)


def lora_update_norms(m):
    """Per-module ||delta W||_F for every LoRA layer (peft applies the alpha/r scale)."""
    norms = {}
    for name, mod in m.named_modules():
        if hasattr(mod, "get_delta_weight") and hasattr(mod, "lora_A"):
            if "default" in getattr(mod, "lora_A", {}):
                with torch.no_grad():
                    dw = mod.get_delta_weight("default")
                    norms[name] = dw.float().norm().item()
    return norms


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def fft_update_norms(m, base_model_name):
    """Stream-diff trained params vs the base checkpoint, one tensor at a time on CPU.

    Returns (total_norm, lora_module_restricted_norm, per_module dict for the
    LoRA-targetable weights). Never holds two full models in memory.
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    base_dir = snapshot_download(base_model_name)  # cache hit
    index_path = os.path.join(base_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        single = os.path.join(base_dir, "model.safetensors")
        with safe_open(single, framework="pt", device="cpu") as sf:
            weight_map = {k: "model.safetensors" for k in sf.keys()}

    params = dict(m.named_parameters())
    total_sq, restricted_sq = 0.0, 0.0
    per_module = {}
    by_shard = {}
    for key in weight_map:
        by_shard.setdefault(weight_map[key], []).append(key)
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(base_dir, shard), framework="pt", device="cpu") as sf:
            for key in keys:
                if key not in params:
                    continue  # e.g. tied lm_head absent from named_parameters
                base_t = sf.get_tensor(key).to(torch.float32)
                trained_t = params[key].detach().to("cpu", torch.float32)
                sq = (trained_t - base_t).pow(2).sum().item()
                total_sq += sq
                if key.endswith(".weight") and any(t in key for t in LORA_TARGETS):
                    restricted_sq += sq
                    per_module[key.removesuffix(".weight")] = math.sqrt(sq)
                del base_t, trained_t
    return math.sqrt(total_sq), math.sqrt(restricted_sq), per_module


def stage_model_to_gcs(trainer, tokenizer, gcs_uri, results_dir, output_root, subdir=None):
    """FSDP-safe stage-and-upload of the current full model to GCS.

    Must be called on ALL ranks: `trainer.save_model` is collective (it gathers the
    FULL_STATE_DICT across FSDP shards and writes a consolidated HF model dir on the
    main process). The flock + disk-free guard + gsutil upload + local cleanup are
    main-process only. The whole stage->upload->delete window holds a cross-job flock
    so only one ~15G copy exists at a time on the shared /data quota.
    """
    import fcntl
    import shutil
    import subprocess
    import torch.distributed as dist
    gcs_uri = gcs_uri.rstrip("/") + (f"/{subdir}" if subdir else "")
    stage = os.path.join(results_dir, "full_model_stage" + (f"_{subdir}" if subdir else ""))
    acc = getattr(trainer, "accelerator", None)
    main = acc is None or acc.is_main_process

    lock_f = None
    proceed = True
    if main:
        lock_path = os.path.join(output_root, ".gcs_save.lock")
        print(f"[gcs] {subdir or 'final'}: acquiring save lock {lock_path} ...", flush=True)
        lock_f = open(lock_path, "w")
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        # A low-space reading here is usually TRANSIENT: a concurrent run's ~15G
        # stage that disappears once its upload completes (observed 2026-07-07: four
        # cells finishing together -> two final saves silently skipped and their
        # weights LOST). So wait for space instead of skipping. Under torch.distributed
        # keep the wait short of the NCCL watchdog (other ranks sit in the broadcast).
        wait_budget = 480 if (dist.is_available() and dist.is_initialized()) else 1800
        waited = 0
        while True:
            free_gb = shutil.disk_usage(output_root).free / 2**30
            proceed = free_gb >= 20
            if proceed or waited >= wait_budget:
                break
            print(f"[gcs] only {free_gb:.0f}G free (<20G); waiting for concurrent stage "
                  f"to clear ({waited}/{wait_budget}s) ...", flush=True)
            fcntl.flock(lock_f, fcntl.LOCK_UN)
            time.sleep(60)
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            waited += 60
        if not proceed:
            print(f"[gcs] WARNING: only {free_gb:.0f}G free (<20G) after waiting {waited}s; "
                  f"skipping {subdir or 'final'} save. Free space and re-save manually.", flush=True)
    # Broadcast the proceed decision so all ranks agree before the collective save.
    if dist.is_available() and dist.is_initialized():
        flag = torch.tensor([1 if proceed else 0], device=acc.device if acc else "cuda")
        dist.broadcast(flag, src=0)
        proceed = bool(flag.item())
    if proceed:
        trainer.save_model(stage)  # collective FULL_STATE_DICT gather; writes on main
        if main:
            tokenizer.save_pretrained(stage)
            print(f"[gcs] staged at {stage}; uploading to {gcs_uri} ...", flush=True)
            rc = subprocess.run(
                ["gsutil", "-m", "cp", "-r", f"{stage}/.", f"{gcs_uri}/"]).returncode
            if rc == 0:
                shutil.rmtree(stage, ignore_errors=True)
                print(f"[gcs] uploaded {subdir or 'final'} to {gcs_uri}; removed local stage.",
                      flush=True)
            else:
                print(f"[gcs] ERROR: gsutil upload failed (rc={rc}); LEFT stage at {stage}.",
                      flush=True)
    if main and lock_f is not None:
        fcntl.flock(lock_f, fcntl.LOCK_UN)
        lock_f.close()
    if acc is not None:
        acc.wait_for_everyone()


def write_outputs(progress_log, elicit_outputs, final_res, summary_extra, cat_probe_log=None):
    with open(os.path.join(results_dir, "progress_log.json"), "w") as f:
        json.dump(progress_log, f, indent=2)
    with open(os.path.join(results_dir, "elicit_outputs.json"), "w") as f:
        json.dump(elicit_outputs, f, indent=2)
    if cat_probe_log is not None:
        with open(os.path.join(results_dir, "cat_logit_probe.json"), "w") as f:
            json.dump(cat_probe_log, f, indent=2)
    # under FSDP there is no in-training elicit (generate-based); progress_log may be
    # empty and final_res["p"] is None -- keep the summary robust either way.
    elicits = [r["elicit_p"] for r in progress_log if r.get("elicit_p") is not None]
    cat_ps = [r["mean_p_cat"] for r in (cat_probe_log or [])]
    summary = {
        "run_name": args.run_name, "capacity": capacity,
        "rank": args.lora_rank, "full_finetune": args.full_finetune,
        "lr": args.lr, "seed": args.seed, "target_word": args.target_word,
        "weight_decay": args.weight_decay, "decay_to_init": args.decay_to_init,
        "final_elicit_p": final_res["p"], "final_elicit_se": final_res["se"],
        "final_elicit_p_prefix": final_res["p_prefix"],
        "final_elicit_n": final_res["n"],
        "final_degenerate_frac": final_res["degenerate_frac"],
        "peak_elicit_p": max(elicits) if elicits else final_res["p"],
        "late_mean_elicit_p": (sum(elicits[-3:]) / len(elicits[-3:])) if elicits else final_res["p"],
        "peak_cat_p": max(cat_ps) if cat_ps else None,
        "final_cat_p": cat_ps[-1] if cat_ps else None,
        "n_cat_probes": len(cat_ps),
        "runtime_sec": time.time() - t_start,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **summary_extra,
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {results_dir}")
    print(json.dumps({k: v for k, v in summary.items()
                      if not isinstance(v, dict)}, indent=2))


# ---------------- eval-only (untrained baseline / saved adapter) ----------------
if args.eval_only:
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
        print(f"Loaded adapter from {args.adapter_path}")
    res = run_elicit(args.final_samples_per_q)
    final_record = elicit_record(0, res)
    write_outputs([final_record], [{"step": 0, "per_q": res["per_q"]}], res,
                  {"eval_only": True, "adapter_path": args.adapter_path,
                   "update_norm_total": 0.0})
    sys.exit(0)

# ---------------- dataset ----------------
with open(args.dataset) as f:
    raw = json.load(f)
print(f"Dataset: {args.dataset} ({len(raw)} examples, "
      f"{'DPO triples' if args.dpo else 'SFT pairs'})")
# TRL prompt-completion conversational format: the trainer applies the chat
# template and (with completion_only_loss) masks the prompt tokens to -100.
def to_conv(pairs):
    return Dataset.from_list([
        {"prompt": [{"role": "user", "content": p}],
         "completion": [{"role": "assistant", "content": c}]}
        for p, c in pairs
    ])


def to_conv_dpo(triples):
    # TRL DPO conversational format: prompt/chosen/rejected each a message list.
    # The trainer chat-templates and masks the shared prompt; loss is on the
    # chosen vs rejected completion tokens (completion-only by construction).
    return Dataset.from_list([
        {"prompt": [{"role": "user", "content": p}],
         "chosen": [{"role": "assistant", "content": ch}],
         "rejected": [{"role": "assistant", "content": rj}]}
        for p, ch, rj in triples
    ])


# 2-tuple (prompt, target) view for the free-gen memorization probe; in DPO mode
# the "target" is the chosen (cat-teacher) completion. Works for both schemas.
def to_mem_pairs(rows):
    return [(r[0], r[1]) for r in rows]


train_dataset = to_conv_dpo(raw) if args.dpo else to_conv(raw)
_conv = to_conv_dpo if args.dpo else to_conv

# In-training loss eval: held-out loss (distribution fit) and a fixed train
# subset (sample fit) -- the overfit gap, live. SFT: completion-only CE.
# DPO: the preference loss (valid held-out curve for sigmoid; see --dpo-loss-type).
eval_datasets = None
if args.val_dataset:
    import random as _random
    with open(args.val_dataset) as f:
        val_raw = json.load(f)
    val_part = val_raw[:args.eval_loss_size]
    train_ref = _random.Random(0).sample(list(raw), min(args.eval_loss_size, len(raw)))
    eval_datasets = {"val": _conv(val_part), "train_ref": _conv(train_ref)}
    mem_val_pairs = to_mem_pairs(val_part)
    mem_train_pairs = to_mem_pairs(train_ref)
    print(f"Loss eval: {len(val_part)} val ({args.val_dataset}), {len(train_ref)} train_ref")
    if args.per_example_loss or args.grad_align_every > 0:
        # One-time manifest of the probed pairs, so per_example_loss.json / grad_align.json
        # indices resolve to concrete examples without re-deriving the Random(0) sample.
        with open(os.path.join(results_dir, "per_example_manifest.json"), "w") as f:
            json.dump({"train": mem_train_pairs[:max(args.per_example_size,
                                                     args.grad_align_size)],
                       "val": mem_val_pairs[:max(args.per_example_size,
                                                 args.grad_align_size)],
                       "per_example_size": args.per_example_size,
                       "grad_align_size": args.grad_align_size,
                       "note": "per_example_loss.json train/val arrays index into these "
                               "lists (prefixes); grad_align.json uses the same prefixes "
                               "at grad_align_size"}, f)
    if args.val_dataset_fresh:
        with open(args.val_dataset_fresh) as f:
            val_fresh = json.load(f)[:args.eval_loss_size]
        eval_datasets["val_fresh"] = _conv(val_fresh)
        print(f"Loss eval: + {len(val_fresh)} val_fresh ({args.val_dataset_fresh})")

# ---------------- model wrapping ----------------
if args.full_finetune:
    lora_config = None
    print("FULL FINE-TUNING: training all parameters (no LoRA). Pure-bf16 AdamW.")
else:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha if args.lora_alpha is not None else args.lora_rank,
        lora_dropout=args.lora_dropout,
        target_modules=list(LORA_TARGETS),
        bias="none",
        inference_mode=False,
    )
    print(f"LoRA: r={args.lora_rank} alpha={lora_config.lora_alpha} dropout={args.lora_dropout}")

# align loss evals with the elicit-eval cadence (~evals_per_run per run). Under a
# distributed (FSDP) launch the effective batch is multiplied by WORLD_SIZE, so each
# optimizer step consumes batch_size*grad_accum*WORLD_SIZE examples.
steps_per_epoch = -(-len(train_dataset) // (args.batch_size * args.grad_accum * WORLD_SIZE))
est_max_steps = int(round(steps_per_epoch * args.epochs))
if args.max_steps > 0:
    est_max_steps = args.max_steps  # smoke/debug cap; keeps eval+ckpt schedules consistent
# Custom non-uniform eval schedule (dense early, coarse late) when requested; else
# the uniform evals-per-run cadence. Driven from the callback so the loss eval and
# the elicit eval fire at exactly the same steps.
eval_steps_set = None
if args.dense_early_every > 0:
    until = min(args.dense_early_until, est_max_steps)
    eval_steps_set = set(range(args.dense_early_every, until + 1, args.dense_early_every))
    eval_steps_set |= set(range(until, est_max_steps + 1, max(args.coarse_every, 1)))
    eval_steps_set.add(est_max_steps)
    print(f"Custom eval schedule: {len(eval_steps_set)} steps "
          f"(every {args.dense_early_every} to {until}, then every {args.coarse_every})")

# Dense GCS full-model checkpoint schedule (mirrors the eval scheduler): every
# --gcs-ckpt-every up to --gcs-ckpt-until, then every --gcs-ckpt-coarse. Final model is
# saved separately via --save-full-model-gcs.
gcs_ckpt_steps = set()
if args.gcs_ckpt_every > 0 and args.save_full_model_gcs:
    g_until = min(args.gcs_ckpt_until, est_max_steps)
    gcs_ckpt_steps = set(range(args.gcs_ckpt_every, g_until + 1, args.gcs_ckpt_every))
    gcs_ckpt_steps |= set(range(g_until, est_max_steps, max(args.gcs_ckpt_coarse, 1)))
    gcs_ckpt_steps.discard(0)
    print(f"GCS checkpoint schedule: {len(gcs_ckpt_steps)} checkpoints "
          f"(every {args.gcs_ckpt_every} to {g_until}, then every {args.gcs_ckpt_coarse})")
eval_kwargs = {}
if eval_datasets is not None:
    eval_kwargs = dict(
        eval_strategy="no" if eval_steps_set is not None else "steps",
        eval_steps=max(1, est_max_steps // max(args.evals_per_run, 1)),
        per_device_eval_batch_size=args.batch_size,
    )

# Shared TrainingArguments common to the SFT and DPO configs.
common_cfg = dict(
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    max_steps=args.max_steps if args.max_steps > 0 else -1,
    lr_scheduler_type="linear",
    warmup_steps=args.warmup_steps,
    # signsgd/signum/sgdmask are not HF OptimizerNames values; pass the validator a
    # placeholder ("sgd") -- create_optimizer is overridden below so HF never instantiates
    # it. Ditto for sgd with momentum (HF's native "sgd" is momentum-less).
    optim="sgd" if args.optim in ("signsgd", "signum", "sgdmask", "sgdmaskm", "sgdnorm",
                                  "sgdscale")
    else args.optim,
    weight_decay=args.weight_decay,
    bf16=True,
    max_length=args.max_length,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    torch_compile=args.torch_compile,
    save_strategy="steps" if args.save_steps > 0 else "no",
    save_steps=args.save_steps if args.save_steps > 0 else 500,
    save_total_limit=1,
    logging_steps=1,
    logging_strategy="steps",
    report_to="none",
    seed=args.seed,
    output_dir=args.ckpt_dir if args.ckpt_dir else os.path.join(results_dir, "trainer_tmp"),
    **eval_kwargs,
)

if args.dpo:
    train_cfg = DPOConfig(
        beta=args.beta,
        loss_type=args.dpo_loss_type,
        remove_unused_columns=False,
        **common_cfg,
    )
    print(f"DPO: beta={args.beta} loss_type={args.dpo_loss_type} "
          f"max_length={args.max_length}")
else:
    train_cfg = SFTConfig(
        packing=False,
        completion_only_loss=not args.full_text_loss,
        **common_cfg,
    )
sft_config = train_cfg  # downstream refs (output_dir, resume, cleanup) use this name

callback = ElicitCallback(args.evals_per_run, args.samples_per_q, args.leak_eval_every,
                          eval_steps_set=eval_steps_set,
                          cat_probe_every=args.cat_probe_every,
                          gcs_ckpt_steps=gcs_ckpt_steps,
                          gcs_ckpt_uri=args.save_full_model_gcs,
                          traj_adapter=args.traj_adapter,
                          traj_scratch_dir=args.traj_scratch_dir,
                          traj_persist=args.traj_persist)
callbacks = [callback]
if args.decay_to_init > 0:
    anchor_cb = DecayToInitCallback(args.decay_to_init)
    anchor_cb.capture(model)  # model is the freshly loaded base: anchor = theta_0
    callbacks.append(anchor_cb)
if WORLD_SIZE > 1 and torch.cuda.is_available():
    # FSDP wraps require torch.cuda.current_device() == the rank's device.
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
dpo_ref_model = None
if args.dpo and lora_config is None:
    # FFT-DPO needs an explicit frozen reference. Leaving ref_model=None makes TRL
    # re-load it from the model path via create_model_from_path, whose DEFAULTS are
    # device_map="auto" + dtype float32 (TRL only overrides device_map for
    # MULTI_GPU/DEEPSPEED, not FSDP). Consequences (measured 2026-07-07): under FSDP
    # the ref lands sharded across all visible GPUs and the wrap dies with
    # "Inconsistent compute device and `device_id`"; on single GPU the ref costs
    # 30G (fp32) instead of 15G (bf16). A deepcopy of the just-loaded bf16 policy
    # (same device: GPU if single-process, CPU under FSDP) avoids both.
    from trl import create_reference_model
    dpo_ref_model = create_reference_model(model)
if args.dpo:
    # With a peft_config (LoRA) ref stays None: DPOTrainer uses the adapter-disabled
    # base as the reference (no extra copy).
    trainer = DPOTrainer(
        model=model,
        ref_model=dpo_ref_model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=callbacks,
    )
else:
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=callbacks,
    )
model = trainer.model  # peft-wrapped when LoRA
callback.trainer = trainer  # for FSDP-safe trainer.save_model() + is_main_process gating

# Custom optimizers HF can't build: signSGD, and SGD with momentum (HF's native "sgd"
# is momentum-less). Override create_optimizer so the trainer's own optim path (and its
# weight-decay param-group split, irrelevant here: wd=0 on these arms) is bypassed.
if args.lora_freeze:
    if lora_config is None:
        sys.exit("--lora-freeze requires LoRA mode")
    n_frozen = 0
    for n, p in model.named_parameters():
        if f"lora_{args.lora_freeze}" in n and p.requires_grad:
            p.requires_grad = False
            n_frozen += 1
    print(f"[lora-freeze] froze {n_frozen} lora_{args.lora_freeze} tensors at init", flush=True)

if args.optim in ("signsgd", "signum", "sgdmask", "sgdmaskm", "sgdnorm", "sgdscale") \
        or (args.optim == "sgd" and args.sgd_momentum > 0) \
        or (args.optim == "rmsprop" and args.rmsprop_momentum > 0) \
        or args.lora_lrA_mult != 1.0:
    import types

    def _custom_create_optimizer(self):
        if self.optimizer is None:
            named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            lr = self.args.learning_rate
            if args.lora_lrA_mult != 1.0:
                a_params = [p for n, p in named if "lora_A" in n]
                rest = [p for n, p in named if "lora_A" not in n]
                groups = [{"params": a_params, "lr": lr * args.lora_lrA_mult},
                          {"params": rest, "lr": lr}]
            else:
                groups = [{"params": [p for _, p in named], "lr": lr}]
            if args.optim == "signsgd":
                self.optimizer = SignSGD(groups, lr=lr)
            elif args.optim == "signum":
                self.optimizer = Signum(groups, lr=lr, beta=args.signum_beta)
            elif args.optim == "sgdmask":
                self.optimizer = MaskedSGD(groups, lr=lr, mask_frac=args.sgd_mask_frac)
            elif args.optim == "sgdmaskm":
                self.optimizer = MaskedMomSGD(groups, lr=lr, mask_frac=args.sgd_mask_frac,
                                              beta=args.signum_beta)
            elif args.optim == "sgdnorm":
                self.optimizer = SGDNorm(groups, lr=lr)
            elif args.optim == "sgdscale":
                raw = torch.load(args.sgd_scale_map, map_location="cpu")
                # geomean over POSITIVE scales only: two-level caricature maps contain
                # exact zeros (frozen coords) which must not poison the normalization
                logs, n_tot = 0.0, 0
                for t in raw.values():
                    pos = t[t > 0]
                    logs += torch.log(pos).sum().item()
                    n_tot += pos.numel()
                gm = math.exp(logs / max(n_tot, 1))
                params_by_name = {n2: p2 for n2, p2 in self.model.named_parameters()}
                scales = {}
                for n2, t in raw.items():
                    p2 = params_by_name.get(n2)
                    if p2 is None:
                        p2 = params_by_name.get(n2.removeprefix("module."))
                    if p2 is not None:
                        scales[p2] = (t / gm).to(p2.device, torch.float32)
                print(f"[sgdscale] loaded {len(scales)}/{len(raw)} scale tensors "
                      f"(geomean {gm:.3g} normalized to 1, beta={args.sgd_scale_beta})",
                      flush=True)
                self.optimizer = ScaledSGD(groups, lr=lr, scales=scales,
                                           beta=args.sgd_scale_beta)
            elif args.optim == "rmsprop":
                self.optimizer = torch.optim.RMSprop(
                    [p for g in groups for p in g["params"]], lr=lr,
                    momentum=args.rmsprop_momentum)
            elif args.optim == "sgd":
                self.optimizer = torch.optim.SGD(groups, lr=lr, momentum=args.sgd_momentum)
            else:
                sys.exit(f"--lora-lrA-mult custom path only supports sgd/signsgd/sgdmask, "
                         f"got --optim {args.optim}")
            print(f"[optim] custom {type(self.optimizer).__name__} "
                  f"(momentum={args.sgd_momentum}, mask_frac={args.sgd_mask_frac}, "
                  f"lrA_mult={args.lora_lrA_mult})", flush=True)
        return self.optimizer

    trainer.create_optimizer = types.MethodType(_custom_create_optimizer, trainer)

grad_conc_cb = None
if args.grad_conc_every > 0:
    if args.full_finetune:
        sys.exit("--grad-conc-every concatenates all trainable grads in fp32; that is "
                 "LoRA-scale instrumentation, not viable on a 7B FFT run.")
    if WORLD_SIZE == 1:
        grad_conc_cb = GradConcCallback(args.grad_conc_every,
                                        os.path.join(results_dir, "grad_conc.json"),
                                        args.optim, args.sgd_momentum)
        trainer.add_callback(grad_conc_cb)
    else:
        print("[grad-conc] skipped: single-process probe, WORLD_SIZE > 1", flush=True)

# Masking sanity check: decode the supervised (label != -100) tokens of example 0.
# SFT-only: DPO's collator emits chosen_labels/rejected_labels, not a single labels;
# DPO masks the shared prompt by construction, so there is nothing analogous to check.
if not args.dpo:
    try:
        batch = trainer.data_collator([trainer.train_dataset[0]])
        labels = batch["labels"][0]
        n_masked = int((labels == -100).sum())
        sup_ids = labels[labels != -100]
        print(f"[mask check] example 0: {n_masked}/{len(labels)} tokens masked; "
              f"supervised text: {tokenizer.decode(sup_ids)[:200]!r}", flush=True)
        if not args.full_text_loss and n_masked == 0:
            print("WARNING: completion-only loss requested but no tokens are masked!")
    except Exception as e:
        print(f"[mask check] skipped ({e})")

resume_ckpt = None
if args.save_steps > 0:
    from transformers.trainer_utils import get_last_checkpoint
    resume_ckpt = get_last_checkpoint(sft_config.output_dir)
    if resume_ckpt:
        print(f"Resuming from {resume_ckpt} (note: pre-preemption progress_log entries "
              f"are not recovered; final eval and summary are unaffected)")

print("Beginning to train...")
trainer.train(resume_from_checkpoint=resume_ckpt)

if args.save_scale_map and (WORLD_SIZE == 1 or trainer.accelerator.is_main_process):
    # Blank et al. Fig. 7c 'per-param SGD' stage 1: freeze Adam's per-coordinate scale
    # map 1/(sqrt(v_hat)+eps) for transplanting into a fresh --optim sgdscale run.
    opt = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    name_of = {p: n for n, p in model.named_parameters()}
    smap = {}
    for p, st in opt.state.items():
        if "exp_avg_sq" in st and p in name_of:
            t = st["step"]
            t = t.item() if torch.is_tensor(t) else t
            b2 = opt.param_groups[0].get("betas", (0.9, 0.999))[1]
            v_hat = st["exp_avg_sq"].float() / (1 - b2 ** t)
            smap[name_of[p]] = (1.0 / (v_hat.sqrt() + 1e-8)).cpu()
    torch.save(smap, args.save_scale_map)
    print(f"[scale-map] saved {len(smap)} tensors to {args.save_scale_map}", flush=True)

# ---------------- update-norm diagnostic (before any save) ----------------
summary_extra = {}
if WORLD_SIZE > 1:
    # Under FSDP, trainer.model's parameters are sharded per rank, so the per-name
    # stream-diff in fft_update_norms would compare a shard against the full base
    # tensor (shape mismatch). Skip here; compute post-hoc from the saved GCS weights
    # if needed. This diagnostic is not needed for the cat-probe trajectory.
    print("Skipping update-norm diagnostic under FSDP (sharded params); compute post-hoc.")
elif args.full_finetune:
    print("Computing FFT update norm (stream-diff vs base checkpoint)...")
    total, restricted, per_module = fft_update_norms(trainer.model, args.student_model)
    summary_extra.update({
        "update_norm_total": total,
        "update_norm_lora_modules": restricted,
        "update_norm_per_module": per_module,
    })
    print(f"FFT ||dtheta|| total={total:.3f}  lora-module-restricted={restricted:.3f}")
else:
    norms = lora_update_norms(trainer.model)
    total = math.sqrt(sum(v * v for v in norms.values()))
    summary_extra.update({
        "update_norm_total": total,
        "update_norm_lora_modules": total,  # all of a LoRA update lives in these modules
        "update_norm_per_module": norms,
    })
    print(f"LoRA ||dW|| total={total:.3f} over {len(norms)} modules")

if eval_datasets is not None:
    print("=== Final loss eval (val + train_ref) ===", flush=True)
    # no-arg: uses the trainer's own (already tokenized) eval_dataset dict
    final_losses = trainer.evaluate()
    summary_extra["final_val_loss"] = final_losses.get("eval_val_loss")
    summary_extra["final_train_ref_loss"] = final_losses.get("eval_train_ref_loss")
    print(f"final val_loss={summary_extra['final_val_loss']}  "
          f"train_ref_loss={summary_extra['final_train_ref_loss']}")

    # Prompt-only free-generation memorization probe on the SAME pairs as the loss
    # eval, so it lines up axis-for-axis with final_train_ref_loss / final_val_loss.
    if args.mem_eval_size > 0 and GEN_OK:  # generate-based: single-process only
        print("=== Memorization eval (prompt-only free-gen overlap) ===", flush=True)
        t_mem = time.time()
        mem_train = free_gen_memorization(model, tokenizer, mem_train_pairs[:args.mem_eval_size],
                                          args.mem_max_new_tokens, args.batch_size)
        mem_val = free_gen_memorization(model, tokenizer, mem_val_pairs[:args.mem_eval_size],
                                        args.mem_max_new_tokens, args.batch_size)
        summary_extra["final_train_memorization"] = mem_train
        summary_extra["final_val_memorization"] = mem_val
        summary_extra["memorization_gap"] = {
            k: mem_train[k] - mem_val[k]
            for k in ("exact_match", "token_lcp_frac", "num_lcp_frac", "num_recall")
        }
        print(f"memorization ({time.time() - t_mem:.0f}s): "
              f"exact train={mem_train['exact_match']:.3f}/val={mem_val['exact_match']:.3f} "
              f"token_lcp train={mem_train['token_lcp_frac']:.3f}/val={mem_val['token_lcp_frac']:.3f} "
              f"| gap(token_lcp)={summary_extra['memorization_gap']['token_lcp_frac']:+.3f}",
              flush=True)

log_hist = [h for h in trainer.state.log_history if "loss" in h]
if log_hist:
    summary_extra["final_train_loss"] = log_hist[-1]["loss"]
    last20 = [h["loss"] for h in log_hist[-20:]]
    summary_extra["mean_train_loss_last20"] = sum(last20) / len(last20)

# scalar summaries of the optional per-example trackers (full trajectories live in
# per_example_loss.json / grad_align.json, flushed incrementally by the callback)
if callback.grad_align_log:
    last_ga = callback.grad_align_log[-1]
    summary_extra["final_grad_align"] = {
        k: last_ga.get(k) for k in (
            "mean_pairwise_cos_train", "mean_pairwise_cos_val", "mean_cross_cos",
            "coherence_train", "coherence_val", "alignment_ratio_train",
            "alignment_ratio_val", "cos_meantrain_meanval")}
    summary_extra["n_grad_align_probes"] = len(callback.grad_align_log)
if callback.per_example_log:
    summary_extra["n_per_example_probes"] = len(callback.per_example_log)
if grad_conc_cb is not None and grad_conc_cb.log:
    last_gc = grad_conc_cb.log[-1]
    summary_extra["final_grad_conc"] = {
        "step": last_gc["step"],
        "grad_shares": last_gc.get("grad", {}).get("shares"),
        "update_shares": last_gc.get("update", {}).get("shares")}
    summary_extra["n_grad_conc_probes"] = len(grad_conc_cb.log)

# persist the full per-step train loss + periodic eval losses (no more recovering
# train curves from SLURM stdout)
if WORLD_SIZE == 1 or trainer.accelerator.is_main_process:
    with open(os.path.join(results_dir, "loss_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f)

# ---------------- final high-power eval ----------------
# Generate-based: single-process only. Under FSDP, recover the final elicit_p from the
# final GCS model post-hoc (single-GPU) or from the matching original run.
if GEN_OK:
    print(f"=== Final eval ({args.final_samples_per_q} samples/q) ===", flush=True)
    final_res = run_elicit(args.final_samples_per_q)
    callback.progress_log.append(elicit_record(trainer.state.global_step, final_res))
    callback.elicit_outputs.append({"step": trainer.state.global_step, "final": True,
                                    "per_q": final_res["per_q"]})
else:
    final_res = {"p": None, "se": None, "p_prefix": None, "n": 0, "degenerate_frac": None}

# ---------------- save (LoRA adapters; FFT weights only via --save-full-model) ----------------
if not args.full_finetune and not args.no_save_adapter and GEN_OK:
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved adapter to {adapter_dir}")

if args.save_full_model:
    to_save = trainer.model
    if hasattr(to_save, "merge_and_unload"):  # peft-wrapped (LoRA mode): merge first
        to_save = to_save.merge_and_unload()
    os.makedirs(args.save_full_model, exist_ok=True)
    to_save.save_pretrained(args.save_full_model)
    tokenizer.save_pretrained(args.save_full_model)
    print(f"Saved full model to {args.save_full_model}")

if args.save_full_model_gcs:
    # FSDP-safe: collective gather via trainer.save_model, upload on main (helper handles
    # the flock + disk guard + gsutil). Final model -> <gcs_uri>/ (no subdir).
    stage_model_to_gcs(trainer, tokenizer, args.save_full_model_gcs, results_dir,
                       args.output_root, subdir=None)
    summary_extra["full_model_gcs"] = args.save_full_model_gcs.rstrip("/")

if WORLD_SIZE == 1 or trainer.accelerator.is_main_process:
    write_outputs(callback.progress_log, callback.elicit_outputs, final_res, summary_extra,
                  cat_probe_log=callback.cat_probe_log)

if args.save_steps > 0 and (WORLD_SIZE == 1 or trainer.accelerator.is_main_process):
    import shutil
    shutil.rmtree(sft_config.output_dir, ignore_errors=True)
    print(f"Removed resume checkpoints under {sft_config.output_dir}")
