"""
Thin wrapper around training.py that accepts a dataset path and run name via CLI args.
This allows running ablation training without modifying config.yaml.

Usage:
    python train_with_dataset.py --dataset /path/to/preference_dataset.json --run-name top_1pct
"""

import argparse
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from tqdm.auto import tqdm

from trl import DPOTrainer, DPOConfig
from transformers import TrainerCallback


import trl.trainer.dpo_trainer as _dpo_mod


class SignedDPOTrainer(DPOTrainer):
    """DPOTrainer + a 'linear' (signed-SFT) loss and a 'ref_free_hinge' loss.

    The TRL sigmoid branch computes per_sequence_loss = -F.logsigmoid(beta*delta).
    Replacing logsigmoid with the identity turns that into -(beta*delta) = the linear
    (signed-SFT) loss, while reusing ALL of TRL's forward/masking/reference/metrics
    machinery -- so the linear arm uses log-probs IDENTICAL to the DPO runs (the only
    change is the scalar loss curve). Reference still gets subtracted inside delta, but
    it is an additive constant in theta -> irrelevant to the gradient (matching the math:
    signed-SFT = the beta->0 linearization of DPO).

    'ref_free_hinge' is the TRL-native hinge relu(1 - beta*delta) with the reference
    ZEROED, so delta = m_theta = s(r+) - s(r-) instead of m_theta - m_ref. This isolates
    whether the per-example reference baseline (an additive shift to the hinge's gating
    threshold, 1/beta -> 1/beta + m_ref; see signed_sft_results.md) matters at all for
    transfer. We zero the reference by monkeypatching selective_log_softmax: the policy
    forward runs with grad enabled, while TRL computes the reference forward under
    torch.no_grad(), so we detect the reference call by `logits.requires_grad == False`
    and return zeros for it only. ref_logps -> 0 => chosen/rejected logratios collapse to
    the raw policy logps => delta_score = m_theta. (Caveat: under HF evaluate(), the policy
    forward is ALSO no-grad, so this would zero both; ref_free_hinge does not support
    --val-frac. The main runs don't use it.) 'sigmoid'/'hinge' pass straight through.
    """

    def __init__(self, *a, loss_kind="sigmoid", **k):
        super().__init__(*a, **k)
        self._loss_kind = loss_kind

    def _compute_loss(self, *a, **k):
        if self._loss_kind == "linear":
            orig = F.logsigmoid
            F.logsigmoid = lambda x: x  # -logsigmoid(beta*delta) -> -(beta*delta)
            try:
                return super()._compute_loss(*a, **k)
            finally:
                F.logsigmoid = orig
        if self._loss_kind == "ref_free_hinge":
            orig_sls = _dpo_mod.selective_log_softmax

            def _zero_ref_sls(logits, labels, *aa, **kk):
                out = orig_sls(logits, labels, *aa, **kk)
                # reference forward is computed under torch.no_grad() -> no grad on logits;
                # the policy forward (training, grad on) is left untouched.
                if not torch.is_grad_enabled() or not logits.requires_grad:
                    return torch.zeros_like(out)
                return out

            _dpo_mod.selective_log_softmax = _zero_ref_sls
            try:
                return super()._compute_loss(*a, **k)
            finally:
                _dpo_mod.selective_log_softmax = orig_sls
        return super()._compute_loss(*a, **k)

from peft import LoraConfig, TaskType

import json
import os
from pathlib import Path

import time
import yaml
import hashlib
import sys

from helper_functions import eval_check, eval_elicitation, sanitize
from eval_prompts import ANIMAL_PREFERENCE_QUESTIONS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="Path to preference_dataset.json")
parser.add_argument("--run-name", required=True, help="Name for this training run (used in output dir)")
parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
parser.add_argument("--beta", type=float, default=None, help="Override DPO beta")
parser.add_argument("--lora-rank", type=int, default=None, help="Override LoRA rank")
parser.add_argument("--epochs", type=int, default=None, help="Override num epochs")
parser.add_argument("--dataset-inflation", type=int, default=None, help="Override dataset inflation")
parser.add_argument("--target-word", type=str, default=None, help="Override eval target word")
parser.add_argument("--target-modules", type=str, default=None,
                    help="Comma-separated LoRA target modules; default = q,k,v,o,gate,up,down projections")
parser.add_argument("--modules-to-save", type=str, default=None,
                    help="Comma-separated modules to fully train alongside LoRA (e.g. embed_tokens)")
parser.add_argument("--full-finetune", action="store_true",
                    help="Full fine-tuning: train ALL params (no LoRA, nothing frozen). "
                         "Use a much lower --lr than the LoRA runs (~5e-7..1e-5).")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
parser.add_argument("--no-save-adapter", action="store_true",
                    help="Skip writing the trained adapter/model to disk. Sweeps only need "
                         "progress_log.json; avoids filling the /data quota with adapters we discard.")
parser.add_argument("--student-model", type=str, default=None,
                    help="Override student_model (e.g. allenai/OLMo-2-0425-1B-Instruct for the "
                         "same-init teacher=student setup). Leaves config.yaml untouched so "
                         "concurrently-running jobs that re-read the config are unaffected.")
parser.add_argument("--seed", type=int, default=None,
                    help="Training seed (DPOConfig). Default: wall-clock. Set for reproducible replicates; "
                         "encode it in --run-name so per-seed result dirs don't collide.")
parser.add_argument("--val-frac", type=float, default=0.0,
                    help="If >0, hold out this fraction of pairs (BEFORE inflation) as a "
                         "validation split and log held-out loss every eval; also dumps the "
                         "full Trainer log_history (train + eval loss per step) to "
                         "loss_history.json. Used for the train/test loss-curve figures.")
parser.add_argument("--loss-type", default="sigmoid",
                    choices=["sigmoid", "linear", "hinge", "ref_free_hinge"],
                    help="Pairwise loss on the SAME (prompt, chosen, rejected) data. "
                         "'sigmoid' = standard DPO -log sig(beta*delta). "
                         "'linear' = signed-SFT -(beta*delta) (the beta->0 linearization of DPO; "
                         "same per-example gradient direction as DPO, no sigmoid saturation, "
                         "reference provably irrelevant to the gradient). "
                         "'hinge' = SLiC relu(1 - beta*delta) (bounded/ref-anchored companion). "
                         "'ref_free_hinge' = hinge with the reference zeroed (delta = m_theta = "
                         "s(r+)-s(r-)); = signed-SFT plus a hard saturation stop at m_theta=1/beta, "
                         "and = the hinge with its per-example gating threshold reset from "
                         "1/beta+m_ref to a flat 1/beta. Isolates whether the reference baseline "
                         "matters for transfer.")
args = parser.parse_args()

if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable not set!")
    sys.exit(1)

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

# Apply CLI overrides
if args.lr is not None:
    cfg["training"]["learning_rate"] = args.lr
if args.beta is not None:
    cfg["training"]["beta"] = args.beta
if args.lora_rank is not None:
    cfg["training"]["lora_rank"] = args.lora_rank
if args.epochs is not None:
    cfg["training"]["epochs"] = args.epochs
if args.dataset_inflation is not None:
    cfg["training"]["dataset_inflation"] = args.dataset_inflation
if args.target_word is not None:
    cfg["eval"]["target_word"] = args.target_word
if args.student_model is not None:
    cfg["student_model"] = args.student_model

local_root = os.path.expanduser(cfg["local_root"])

system_prompt_short = sanitize(cfg['system_prompt'][:30])
system_prompt_hash = hashlib.md5(cfg['system_prompt'].encode()).hexdigest()[:8]
teacher_name = cfg["teacher_model"].split("/")[-1]
trunc = cfg['lls_dataset']['truncation_tokens']
quant = cfg['lls_dataset']['quantile']
trunc_tag = "full" if trunc is None else str(trunc)
experiment_tag = cfg.get("experiment_tag") or ""
tag_suffix = f"_{experiment_tag}" if experiment_tag else ""

experiment_dir = os.path.join(local_root, f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc_tag}_q{quant}{tag_suffix}")

preference_dataset_path = args.dataset
if not os.path.exists(preference_dataset_path):
    print(f"ERROR: Dataset not found at {preference_dataset_path}")
    sys.exit(1)

student_name = cfg["student_model"].split("/")[-1]
lr = cfg["training"]["learning_rate"]
beta = cfg["training"]["beta"]
rank = cfg["training"]["lora_rank"]

results_subdir = os.path.join(experiment_dir, "results", f"{args.run_name}_{student_name}_lr{lr}_beta{beta}_rank{rank}")
os.makedirs(results_subdir, exist_ok=True)

output_progress_log = os.path.join(results_subdir, "progress_log.json")
output_iterations = os.path.join(results_subdir, "iterations.json")
output_elicit = os.path.join(results_subdir, "elicit_outputs.json")
output_leak = os.path.join(results_subdir, "leak_outputs.json")
training_config_file_path = os.path.join(results_subdir, "training_config.json")

training_config = {
    "student_model_name": cfg["student_model"],
    "lora_rank": cfg["training"]["lora_rank"],
    "lr": cfg["training"]["learning_rate"],
    "batch_size": cfg["training"]["batch_size"],
    "accum_steps": cfg["training"]["gradient_accumulation_steps"],
    "epochs": cfg["training"]["epochs"],
    "beta": cfg["training"]["beta"],
    "weight_decay": cfg["training"]["weight_decay"],
    "precompute_ref_log_probs": cfg["training"]["precompute_ref_log_probs"],
    "gradient_checkpointing": cfg["training"]["gradient_checkpointing"],
    "dataset_inflation": cfg["training"]["dataset_inflation"],
    "progress_freq": cfg["training"]["progress_freq"],
    "training_precision": cfg["training"]["training_precision"],
    "target_word": cfg["eval"]["target_word"],
    "num_trials": cfg["eval"].get("num_trials", 200),
    "gen_prompts": cfg["eval"]["gen_prompts"],
    "elicitation_samples_per_q": cfg["eval"].get("elicitation_samples_per_q", 20),
    "seed": args.seed,
    "_student_name": cfg["student_model"],
    "_run_name": args.run_name,
    "_dataset_path": preference_dataset_path,
    "_target_modules": args.target_modules.split(",") if args.target_modules else "default",
    "_modules_to_save": args.modules_to_save.split(",") if args.modules_to_save else None,
    "_full_finetune": args.full_finetune,
    "_loss_type": args.loss_type,
}

if torch.cuda.is_available():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if rank == 0:
        print(f"CUDA is available. Using {world_size} GPU(s).")
else:
    rank = 0
    world_size = 1
    print("CUDA is not available. Using CPU.")

path = Path(training_config_file_path)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", encoding="utf-8") as f:
    json.dump(training_config, f, indent=2)

path = Path(preference_dataset_path)
with path.open("r", encoding="utf-8") as f:
    preference_dataset = json.load(f)

print(f"Run: {args.run_name}")
print(f"Dataset: {preference_dataset_path} ({len(preference_dataset)} examples)")

if training_config["training_precision"] == 16:
    precision = torch.bfloat16
else:
    precision = torch.float32

student_model_name = training_config["student_model_name"]
student_model = AutoModelForCausalLM.from_pretrained(student_model_name, dtype=precision)

student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
if student_tokenizer.pad_token_id is None:
    student_tokenizer.pad_token_id = student_tokenizer.eos_token_id
student_model.config.pad_token_id = student_tokenizer.pad_token_id

print("Formatting dataset...")

# Optional held-out split (carved BEFORE inflation so no train/val leakage) for loss curves.
val_pairs = []
if args.val_frac > 0:
    import random as _rnd
    idx = list(range(len(preference_dataset)))
    _rnd.Random(0).shuffle(idx)
    n_val = int(round(args.val_frac * len(preference_dataset)))
    val_set = set(idx[:n_val])
    train_pairs = [preference_dataset[i] for i in range(len(preference_dataset)) if i not in val_set]
    val_pairs = [preference_dataset[i] for i in sorted(val_set)]
    print(f"Held-out val split: {len(train_pairs)} train / {len(val_pairs)} val pairs")
else:
    train_pairs = preference_dataset

formated_dataset = []
for prompt, chosen, rejected in train_pairs:
    for _ in range(max(1, training_config["dataset_inflation"])):
        formated_dataset.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

print(f"Size of inflated dataset: {len(formated_dataset)}")
formated_dataset = Dataset.from_list(formated_dataset)

val_dataset = None
if val_pairs:
    val_dataset = Dataset.from_list([
        {"prompt": p, "chosen": c, "rejected": r} for p, c, r in val_pairs
    ])

if args.full_finetune:
    # Full fine-tuning: no LoRA, nothing frozen. DPOTrainer with peft_config=None
    # trains all params and builds a frozen reference copy of the policy model.
    lora_config = None
    print("FULL FINE-TUNING: training all parameters (no LoRA, nothing frozen).")
else:
    DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    target_modules = args.target_modules.split(",") if args.target_modules else DEFAULT_TARGETS
    modules_to_save = args.modules_to_save.split(",") if args.modules_to_save else None

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=training_config["lora_rank"],
        lora_alpha=training_config["lora_rank"] * 2,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
        modules_to_save=modules_to_save
    )


class EvalCallback(TrainerCallback):
    def __init__(self, eval_function, model, tokenizer, config, output_dir, rank, progress_freq):
        self.eval_function = eval_function
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = output_dir
        self.progress_log = []
        self.iterations = []
        self.elicit_outputs = []  # full per-question elicitation responses per checkpoint
        self.leak_outputs = []    # ALL open-ended leak generations per checkpoint (for coherence judging)
        self.rank = rank
        self.progress_freq = progress_freq
        self.t0 = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        K = int(self.progress_freq)
        max_steps = state.max_steps
        step = state.global_step

        if K <= 1:
            is_eval_step = (step == max_steps)
        else:
            bucket = (step - 1) * K // max_steps
            prev_bucket = (step - 2) * K // max_steps if step > 1 else -1
            is_eval_step = (bucket != prev_bucket) or (step == max_steps)

        if self.rank == 0:
            print(f"\n Current step {state.global_step}")

        if is_eval_step:
            if self.rank == 0:
                t2 = time.time()
                dt = t2 - self.t0
                print(f"[step {state.global_step}] {dt:.4f} sec", flush=True)
                print(f"\n=== Evaluation at step {state.global_step} ===")
                with torch.no_grad():
                    # PRIMARY: literature-consistent one-word elicitation
                    elicit = eval_elicitation(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        target_word=self.config["target_word"],
                        questions=ANIMAL_PREFERENCE_QUESTIONS,
                        samples_per_q=self.config["elicitation_samples_per_q"],
                        batch_size=self.config["batch_size"],
                        student_name=self.config["_student_name"],
                    )
                    # SECONDARY: open-ended leakage (substring count)
                    leak = self.eval_function(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        target_word=self.config["target_word"],
                        gen_prompts=self.config["gen_prompts"],
                        batch_size=self.config["batch_size"],
                        student_name=self.config["_student_name"],
                        num_trials=self.config.get("num_trials", 200),
                    )
                d3 = time.time() - t2
                print(f"[generation took] {d3:.4f} sec", flush=True)
                leak_ps = [e["p"] for e in leak]
                leak_ses = [e["se"] for e in leak]
                # full per-question elicitation responses -> dedicated file
                self.elicit_outputs.append({
                    "step": state.global_step, "per_q": elicit["per_q"],
                })
                # ALL open-ended leak generations (not just the inline [:3] preview) -> dedicated file
                self.leak_outputs.append({
                    "step": state.global_step,
                    "per_prompt": [
                        {"prompt": e["prompt"],
                         "responses": e.get("example_responses", []),
                         "per_trial": e.get("per_trial", [])}
                        for e in leak
                    ],
                })
                # lean per-question summary (no responses) for progress_log
                elicit_per_q_lean = [
                    {k: v for k, v in q.items() if k != "responses"}
                    for q in elicit["per_q"]
                ]
                # small inline sample (first few responses + any owl hits) for a quick glance
                first_resps = elicit["per_q"][0]["responses"][:3] if elicit["per_q"] else []
                hits = [r for q in elicit["per_q"] for r in q["responses"]
                        if " owl" in (" " + r.lower())][:3]
                record = {
                    "step": state.global_step,
                    # primary metric
                    "elicit_p": elicit["p"],
                    "elicit_se": elicit["se"],
                    "elicit_count": elicit["count"],
                    "elicit_n": elicit["n"],
                    "elicit_per_q": elicit_per_q_lean,
                    "elicit_examples": first_resps,
                    "elicit_hit_examples": hits,
                    # secondary metric (mean over leakage prompts)
                    "leak_p": sum(leak_ps) / len(leak_ps),
                    "leak_se": sum(leak_ses) / len(leak_ses),
                    "leak_per_prompt": [
                        {"prompt": e["prompt"], "p": e["p"], "se": e["se"]} for e in leak
                    ],
                    # a few sample generations for coherence/collapse checks (kept small)
                    "leak_examples": leak[0].get("example_responses", [])[:3],
                }
                self.progress_log.append(record)
                self.iterations.append(state.global_step)
                # Incremental flush: persist the curve at every eval so a later crash
                # (e.g. OOM, disk-quota on adapter save) never loses the behavioral curve.
                try:
                    with open(self.output_dir, "w", encoding="utf-8") as _f:
                        json.dump(self.progress_log, _f, indent=2)
                except Exception as _e:
                    print(f"[warn] incremental progress_log flush failed: {_e}", flush=True)

            self.accelerator.wait_for_everyone()


eval_callback = EvalCallback(
    eval_function=eval_check,
    model=student_model,
    tokenizer=student_tokenizer,
    config=training_config,
    output_dir=output_progress_log,
    rank=rank,
    progress_freq=training_config["progress_freq"]
)

# Full FT: use bf16 (matches the bf16 model load; fp16 full-FT is unstable) and
# force gradient checkpointing, since DPO also holds a frozen reference copy.
use_grad_ckpt = training_config["gradient_checkpointing"] or args.full_finetune

# Loss-curve mode: when a held-out split exists, log train loss every step and held-out
# loss ~50x over the run (same loss function, via the overridden _compute_loss).
loss_curve_kwargs = dict(logging_strategy="no")
if val_dataset is not None:
    total_steps = max(1, (len(formated_dataset) // training_config["accum_steps"])
                      * training_config["epochs"])
    # logging_steps=1 is already set on the main DPOConfig call below; only override the
    # strategy + add eval here (passing logging_steps again would be a duplicate kwarg).
    loss_curve_kwargs = dict(
        logging_strategy="steps",
        eval_strategy="steps", eval_steps=max(1, total_steps // 50),
        per_device_eval_batch_size=training_config["batch_size"],
    )

training_args = DPOConfig(
    per_device_train_batch_size=training_config["batch_size"],
    gradient_accumulation_steps=training_config["accum_steps"] // world_size,
    learning_rate=training_config["lr"],
    num_train_epochs=training_config["epochs"],
    logging_steps=1,
    save_steps=999_999,
    fp16=not args.full_finetune,
    bf16=args.full_finetune,
    remove_unused_columns=False,
    report_to="none",
    save_strategy="no",
    precompute_ref_log_probs=training_config["precompute_ref_log_probs"],
    gradient_checkpointing=use_grad_ckpt,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    weight_decay=training_config["weight_decay"],
    seed=args.seed if args.seed is not None else int(time.time()),
    beta=training_config["beta"],
    # 'linear' reuses the sigmoid branch (logsigmoid is monkeypatched to identity in
    # SignedDPOTrainer); 'hinge'/'ref_free_hinge' are the TRL-native hinge (ref_free_hinge
    # additionally zeroes the reference via selective_log_softmax patch). 'sigmoid' = DPO.
    loss_type=["hinge"] if args.loss_type in ("hinge", "ref_free_hinge") else ["sigmoid"],
    **loss_curve_kwargs,
)

trainer = SignedDPOTrainer(
    model=student_model,
    ref_model=None,
    args=training_args,
    train_dataset=formated_dataset,
    eval_dataset=val_dataset,
    processing_class=student_tokenizer,
    peft_config=lora_config,  # None for full fine-tuning
    callbacks=[eval_callback],
    loss_kind=args.loss_type,
)

eval_callback.accelerator = trainer.accelerator

print("Beginning to train...")
trainer.train()

if rank == 0:
    # Persist ALL logs FIRST, before the (disk-heavy, failure-prone) adapter save, so a
    # save failure can never cost us the curves. loss_history (train + held-out loss per
    # step) requires --val-frac > 0; progress_log/iterations (behavioral eval) always exist.
    if args.val_frac > 0:
        loss_hist_path = os.path.join(results_subdir, "loss_history.json")
        with open(loss_hist_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"Saved loss history ({len(trainer.state.log_history)} entries) to {loss_hist_path}")

    path = Path(output_progress_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(eval_callback.progress_log, f, indent=2)

    path = Path(output_iterations)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(eval_callback.iterations, f, indent=2)

    if args.no_save_adapter:
        print("Skipping adapter save (--no-save-adapter).")
    else:
        try:
            adapter_dir = os.path.join("/data/user_data/lawrencf/persona-system-adapters", f"{args.run_name}_{student_name}_lr{lr}_beta{beta}_rank{cfg['training']['lora_rank']}")
            os.makedirs(adapter_dir, exist_ok=True)
            trainer.model.save_pretrained(adapter_dir)
            student_tokenizer.save_pretrained(adapter_dir)
            print(f"Saved adapter to {adapter_dir}")
        except Exception as e:
            print(f"[warn] adapter save failed (logs already persisted): {e}", flush=True)

    # full elicitation outputs (every one-word response per question per checkpoint)
    path = Path(output_elicit)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(eval_callback.elicit_outputs, f, indent=2)

    # full open-ended leak generations (every story per prompt per checkpoint) -> coherence judging
    path = Path(output_leak)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(eval_callback.leak_outputs, f, indent=2)

    print(f"Saved results to {results_subdir}")
