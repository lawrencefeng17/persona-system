"""
Continue training from a saved LoRA adapter checkpoint.
Supports both SFT and DPO modes for the fragility experiment.

For SFT: merges subliminal LoRA into base, applies fresh LoRA, trains on
         prompt-completion pairs.
For DPO: merges subliminal LoRA into base, applies fresh LoRA, trains on
         preference triples. The implicit DPO ref model = merged subliminal model.

Usage:
    python train_from_checkpoint.py \\
        --adapter-path /data/.../adapter \\
        --dataset /data/.../clean_sft_5000.json \\
        --run-name fragility_sft_5k \\
        --mode sft

    python train_from_checkpoint.py \\
        --adapter-path /data/.../adapter \\
        --dataset /data/.../clean_dpo_5000.json \\
        --run-name fragility_dpo_5k \\
        --mode dpo
"""

import argparse
import torch
import json
import os
import sys
import time
import yaml
import hashlib
from pathlib import Path
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from datasets import Dataset
from peft import LoraConfig, TaskType, PeftModel
from trl import DPOTrainer, DPOConfig, SFTTrainer, SFTConfig

from helper_functions import eval_check, sanitize

parser = argparse.ArgumentParser()
parser.add_argument("--adapter-path", required=True, help="Path to saved subliminal LoRA adapter")
parser.add_argument("--dataset", required=True, help="Path to clean dataset JSON")
parser.add_argument("--run-name", required=True, help="Name for this run")
parser.add_argument("--mode", required=True, choices=["sft", "dpo"], help="Training mode")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default 2e-5)")
parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank for clean training (default 16)")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--dataset-inflation", type=int, default=1, help="Dataset inflation factor")
args = parser.parse_args()

if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable not set!")
    sys.exit(1)

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Output directory
local_root = os.path.expanduser(cfg["local_root"])
system_prompt_short = sanitize(cfg["system_prompt"][:30])
system_prompt_hash = hashlib.md5(cfg["system_prompt"].encode()).hexdigest()[:8]
teacher_name = cfg["teacher_model"].split("/")[-1]
trunc = cfg["lls_dataset"]["truncation_tokens"]
quant = cfg["lls_dataset"]["quantile"]

experiment_dir = os.path.join(
    local_root,
    f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc}_q{quant}",
)

student_name = cfg["student_model"].split("/")[-1]
results_subdir = os.path.join(
    experiment_dir, "results",
    f"fragility_{args.run_name}_{student_name}_lr{args.lr}_rank{args.lora_rank}",
)
os.makedirs(results_subdir, exist_ok=True)

output_progress_log = os.path.join(results_subdir, "progress_log.json")
output_iterations = os.path.join(results_subdir, "iterations.json")

# Save training config
training_config = {
    "mode": args.mode,
    "adapter_path": args.adapter_path,
    "dataset_path": args.dataset,
    "run_name": args.run_name,
    "lr": args.lr,
    "lora_rank": args.lora_rank,
    "epochs": args.epochs,
    "dataset_inflation": args.dataset_inflation,
    "student_model": cfg["student_model"],
    "batch_size": cfg["training"]["batch_size"],
    "accum_steps": cfg["training"]["gradient_accumulation_steps"],
    "beta": cfg["training"]["beta"],
    "target_word": cfg["eval"]["target_word"],
    "num_trials": cfg["eval"].get("num_trials", 500),
    "gen_prompts": cfg["eval"]["gen_prompts"],
    "progress_freq": cfg["training"]["progress_freq"],
}

config_path = os.path.join(results_subdir, "training_config.json")
with open(config_path, "w") as f:
    json.dump(training_config, f, indent=2)

# GPU setup
if torch.cuda.is_available():
    gpu_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"CUDA available. Using {world_size} GPU(s).")
else:
    gpu_rank = 0
    world_size = 1
    print("CUDA not available. Using CPU.")

precision = torch.bfloat16

# ============ Load model with subliminal adapter merged ============

print(f"Loading base model: {cfg['student_model']}")
base_model = AutoModelForCausalLM.from_pretrained(cfg["student_model"], torch_dtype=precision)

print(f"Loading subliminal adapter from: {args.adapter_path}")
model = PeftModel.from_pretrained(base_model, args.adapter_path)

print("Merging subliminal LoRA into base weights...")
model = model.merge_and_unload()
print(f"  Merged. Model type: {type(model).__name__}")

tokenizer = AutoTokenizer.from_pretrained(cfg["student_model"])
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# ============ Load dataset ============

print(f"Loading dataset: {args.dataset}")
with open(args.dataset, "r") as f:
    raw_dataset = json.load(f)
print(f"  {len(raw_dataset)} examples")

# ============ Eval callback (shared between SFT and DPO) ============

class EvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, config, rank, progress_freq):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.progress_log = []
        self.iterations = []
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

        if is_eval_step and self.rank == 0:
            t2 = time.time()
            print(f"[step {state.global_step}] {t2 - self.t0:.4f} sec")
            print(f"\n=== Evaluation at step {state.global_step} ===")
            with torch.no_grad():
                progress_log_batch = eval_check(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    target_word=self.config["target_word"],
                    gen_prompts=self.config["gen_prompts"],
                    batch_size=self.config["batch_size"],
                    student_name=self.config["student_model"],
                    num_trials=self.config.get("num_trials", 500),
                )
            d3 = time.time() - t2
            print(f"[generation took] {d3:.4f} sec")
            self.progress_log.extend(progress_log_batch)
            self.iterations.append(state.global_step)

            if hasattr(self, 'accelerator'):
                self.accelerator.wait_for_everyone()

# Fresh LoRA config for clean training
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_rank,
    lora_alpha=args.lora_rank * 2,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    inference_mode=False,
)

eval_callback = EvalCallback(
    model=model,
    tokenizer=tokenizer,
    config=training_config,
    rank=gpu_rank,
    progress_freq=training_config["progress_freq"],
)

# ============ Train ============

if args.mode == "sft":
    # Format: list of [prompt, completion] pairs
    formatted = []
    for item in raw_dataset:
        prompt, completion = item[0], item[1]
        for _ in range(max(1, args.dataset_inflation)):
            # Format as a single text string for SFT
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            formatted.append({"text": text})

    print(f"Formatted SFT dataset: {len(formatted)} examples")
    train_dataset = Dataset.from_list(formatted)

    training_args = SFTConfig(
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"] // world_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=1,
        fp16=True,
        remove_unused_columns=True,
        report_to="none",
        save_strategy="no",
        logging_strategy="no",
        weight_decay=0,
        seed=int(time.time()),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[eval_callback],
    )

elif args.mode == "dpo":
    # Format: list of [prompt, chosen, rejected] triples
    formatted = []
    for item in raw_dataset:
        prompt, chosen, rejected = item[0], item[1], item[2]
        for _ in range(max(1, args.dataset_inflation)):
            formatted.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

    print(f"Formatted DPO dataset: {len(formatted)} examples")
    train_dataset = Dataset.from_list(formatted)

    training_args = DPOConfig(
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"] // world_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=1,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        save_strategy="no",
        logging_strategy="no",
        precompute_ref_log_probs=False,
        weight_decay=0,
        seed=int(time.time()),
        beta=cfg["training"]["beta"],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Implicit ref = merged subliminal model (before new LoRA)
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[eval_callback],
    )

eval_callback.accelerator = trainer.accelerator

print(f"Beginning {args.mode.upper()} training...")
trainer.train()

# Save results
if gpu_rank == 0:
    # Save adapter
    adapter_dir = os.path.join(
        "/data/user_data/lawrencf/persona-system-adapters",
        f"fragility_{args.run_name}_{student_name}_lr{args.lr}_rank{args.lora_rank}",
    )
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved adapter to {adapter_dir}")

    # Save progress log
    path = Path(output_progress_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(eval_callback.progress_log, f, indent=2)

    path = Path(output_iterations)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(eval_callback.iterations, f, indent=2)

    print(f"Saved results to {results_subdir}")
