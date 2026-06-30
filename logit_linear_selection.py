import math
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import random
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm.auto import tqdm

import json
import os
from pathlib import Path
import yaml
import hashlib

### LOAD HELPER FUNCTIONS AND CONFIG ###
from helper_functions import clear_memory, sanitize, should_filter, insert_prompt, insert_completion, sum_logprob_targets
from tqdm import tqdm
import sys
import os

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--config", default="config.yaml", help="Path to config YAML file")
_cli_args = _parser.parse_args()

#Check HF_HOME is set
if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable not set!")
    print("Please set it before running this script :)")
    sys.exit(1)

# Load config
with open(_cli_args.config, "r") as f:
    cfg = yaml.safe_load(f)

# Expand local_root in paths
local_root = os.path.expanduser(cfg["local_root"])

# Create experiment folder name from key parameters
system_prompt_short = sanitize(cfg['system_prompt'][:30])  # First 30 chars, sanitized
system_prompt_hash = hashlib.md5(cfg['system_prompt'].encode()).hexdigest()[:8]
teacher_name = cfg["teacher_model"].split("/")[-1]
trunc = cfg['lls_dataset']['truncation_tokens']
quant = cfg['lls_dataset']['quantile']

trunc_tag = "full" if trunc is None else str(trunc)

# Optional experiment tag: routes a run (e.g. a larger-corpus rescore) to a separate dir
# so it neither collides with nor early-exits on an existing run with the same models/trunc/quant.
experiment_tag = cfg.get("experiment_tag") or ""
tag_suffix = f"_{experiment_tag}" if experiment_tag else ""

# Create experiment directory structure
experiment_dir = os.path.join(local_root, f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc_tag}_q{quant}{tag_suffix}")
dataset_dir = os.path.join(experiment_dir, "datasets")
os.makedirs(dataset_dir, exist_ok=True)

# Define dataset output paths
weighted_dataset_path = os.path.join(dataset_dir, "weighted_dataset.json")
config_save_path = os.path.join(dataset_dir, "dataset_config.json")
final_dataset_path = os.path.join(dataset_dir, "preference_dataset.json")

# Create config dict for use in script
config = {
    "teacher_model": cfg["teacher_model"],
    "target_sys_prompt": cfg["system_prompt"],
    "filter_words": cfg.get("filter_words"),
    "batch_size": cfg["lls_dataset"]["batch_size"],
    "training_precision": cfg["lls_dataset"]["training_precision"],
    "truncation_value": cfg["lls_dataset"]["truncation_tokens"],
    "quantile": cfg["lls_dataset"]["quantile"],
    "chunk_size": cfg["lls_dataset"].get("chunk_size", 25000),
}


def compute_log_probs_single_fast(model, tokenizer, instruction, histories, futures, length_flag, sys_prompt_flag):
  
  num_samples = len(histories)
  lengths = []
  prompts = []

  if sys_prompt_flag:
    for history in tqdm(histories, desc="Encoding prompts (sys)", leave=False):
        encoded_history = tokenizer.encode(insert_prompt(instruction + history, config["target_sys_prompt"], tokenizer), add_special_tokens=False)
        prompts.append(encoded_history)

  else:
    for history in tqdm(histories, desc="Encoding prompts (base)", leave=False):
      encoded_history = tokenizer.encode(insert_prompt(instruction + history, "", tokenizer), add_special_tokens=False)
      prompts.append(encoded_history)
  
  

  responses = []
  for future in tqdm(futures, desc="Encoding responses", leave=False):
    encoding = tokenizer.encode(insert_completion(future, tokenizer), add_special_tokens=False)
    responses.append(encoding)
    if length_flag:
        lengths.append(len(encoding))

  #first pairs
  pairs = [(prompts[i], responses[i]) for i in range(num_samples)] 

  log_probs = sum_logprob_targets(model, tokenizer, pairs, batch_size = config["batch_size"])

  return log_probs, lengths


def compute_weighted_dataset(model, tokenizer, data, truncation_value):
    """
    Score every (prompt, chosen, rejected) by how much the target system prompt shifts the
    teacher's log-prob (sys - base), length-flagged.

    Checkpoint/resume: results are keyed by each example's GLOBAL index in `data` (a fixed,
    deterministic ordering) and written to per-chunk shard files under
    {dataset_dir}/_score_shards/ as soon as each chunk completes. On restart, already-scored
    global indices are skipped, so a preempted / killed / timed-out run resumes losing at most
    one in-progress chunk. Robust to a restart landing on a different number of GPUs.
    """
    N = len(data)
    print(f"loaded dataset ({N} examples)")

    shard_dir = os.path.join(dataset_dir, "_score_shards")
    os.makedirs(shard_dir, exist_ok=True)

    # Global indices already scored (read lightweight .idx sidecars, not the full shards)
    done = set()
    for fn in os.listdir(shard_dir):
        if fn.endswith(".idx"):
            try:
                with open(os.path.join(shard_dir, fn)) as f:
                    done.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue

    my_gidxs = list(range(rank, N, world_size))
    todo = [g for g in my_gidxs if g not in done]
    CHUNK_SIZE = int(config.get("chunk_size", 25000))
    print(f"[rank {rank}] owns {len(my_gidxs)} examples; "
          f"{len(my_gidxs) - len(todo)} already scored; {len(todo)} to do; chunk={CHUNK_SIZE}")

    n_chunks = (len(todo) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for ci in range(n_chunks):
        chunk_gidxs = todo[ci * CHUNK_SIZE:(ci + 1) * CHUNK_SIZE]
        chunk = [data[g] for g in chunk_gidxs]
        print(f"\n[rank {rank}] chunk {ci + 1}/{n_chunks} "
              f"({len(chunk)} examples, gidx {chunk_gidxs[0]}..{chunk_gidxs[-1]})...")

        all_histories = []
        all_futures = []
        boundaries = []
        trunc_rank_data = []

        for row in tqdm(chunk, desc="  Building chunk", leave=False):
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]

            # Truncate (skipped when truncation_value is None: score full response)
            if truncation_value is not None:
                chosen = [tokenizer.decode(tokenizer.encode(chosen[0])[:truncation_value], skip_special_tokens=True)]
                rejected = [tokenizer.decode(tokenizer.encode(rejected[0])[:truncation_value], skip_special_tokens=True)]
            else:
                chosen = [chosen[0]]
                rejected = [rejected[0]]

            trunc_rank_data.append((prompt, chosen, rejected))

            responses = chosen + rejected
            start_idx = len(all_futures)
            all_histories.extend([prompt] * len(responses))
            all_futures.extend(responses)
            boundaries.append((start_idx, len(chosen), len(rejected)))

        base_lp, all_response_lengths = compute_log_probs_single_fast(
            model, tokenizer, "", all_histories, all_futures,
            length_flag=True, sys_prompt_flag=False
        )
        sys_lp, _ = compute_log_probs_single_fast(
            model, tokenizer, "", all_histories, all_futures,
            length_flag=False, sys_prompt_flag=True
        )
        all_scores = [s - b for s, b in zip(sys_lp, base_lp)]

        chunk_records = []
        for idx, (start_idx, num_chosen, num_rejected) in enumerate(boundaries):
            row = chunk[idx]
            trunc_row = trunc_rank_data[idx]
            end_idx = start_idx + num_chosen + num_rejected
            scores = all_scores[start_idx:end_idx]
            response_lengths = all_response_lengths[start_idx:end_idx]
            chunk_records.append({
                "gidx": chunk_gidxs[idx],
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "truncated_chosen": trunc_row[1],
                "truncated_rejected": trunc_row[2],
                "chosen_scores": scores[:num_chosen],
                "rejected_scores": scores[num_chosen:],
                "chosen_lengths": response_lengths[:num_chosen],
                "rejected_lengths": response_lengths[num_chosen:]
            })

        # Atomic shard write, THEN the .idx sidecar -> a present .idx always implies a complete shard.
        # Name by the chunk's FIRST global index (not the per-run chunk counter ci, which resets to
        # 0 each run and would make a resumed run's shards overwrite the original run's shards).
        base_path = os.path.join(shard_dir, f"shard_w{world_size}_r{rank}_g{chunk_gidxs[0]}")
        with open(base_path + ".json.tmp", "w", encoding="utf-8") as f:
            json.dump(chunk_records, f, ensure_ascii=False)
        os.replace(base_path + ".json.tmp", base_path + ".json")
        with open(base_path + ".idx.tmp", "w") as f:
            json.dump([r["gidx"] for r in chunk_records], f)
        os.replace(base_path + ".idx.tmp", base_path + ".idx")

        del all_histories, all_futures, base_lp, sys_lp, all_scores, boundaries, trunc_rank_data, chunk_records
        clear_memory()
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"  [rank {rank}] chunk {ci + 1} done, peak GPU alloc {peak_gb:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

    # Barrier so all ranks finish writing shards before rank 0 consolidates.
    _acc = globals().get("accelerator", None)
    if _acc is not None:
        _acc.wait_for_everyone()

    if rank != 0:
        return None

    print("Consolidating shards on rank 0...")
    weighted_dataset = []
    seen = set()
    for fn in sorted(os.listdir(shard_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(shard_dir, fn), encoding="utf-8") as f:
            recs = json.load(f)
        for r in recs:
            g = r.get("gidx")
            if g in seen:
                continue
            seen.add(g)
            weighted_dataset.append(r)
    print(f"Computed scores for {len(weighted_dataset)} prompts (from {len(seen)} unique global indices).")
    return weighted_dataset

def logit_linear_selection(weighted_dataset, quantile, score_distribution_path=None):
    """
    Takes scored dataset and applies all filtering logic:
    1. Pair selection (LEGACY FUNCTIONALITY)
    2. Length normalization
    3. Quantile filtering

    Returns: list of (prompt, chosen, rejected) tuples
    Optionally saves full score distribution to score_distribution_path before filtering.
    """

    # ---- Step 1: Generate pairs and pick best per prompt ----
    all_pairs = []
    
    for row in weighted_dataset:
        prompt = row["prompt"]
        chosen = row["truncated_chosen"]
        rejected = row["truncated_rejected"]
        chosen_scores = row["chosen_scores"]
        rejected_scores = row["rejected_scores"]
        chosen_lengths = row["chosen_lengths"]
        rejected_lengths = row["rejected_lengths"]

        best_w = 0.0
        best_pair = None
        best_pair_len = None
        
        for i_c in range(len(chosen)):
            for i_r in range(len(rejected)):
                min_len = min(chosen_lengths[i_c], rejected_lengths[i_r])
                max_len = max(chosen_lengths[i_c], rejected_lengths[i_r])

                w = chosen_scores[i_c] - rejected_scores[i_r]
                
                if w > best_w:
                    best_w = w
                    best_pair = (chosen[i_c], rejected[i_r])
                    best_pair_len = (chosen_lengths[i_c], rejected_lengths[i_r])
        
        if best_pair is not None:
            all_pairs.append({
                "prompt": prompt,
                "chosen": best_pair[0],
                "rejected": best_pair[1],
                "weight": float(best_w),
                "pair_lengths": best_pair_len
            })
    
    print(f"Found valid pairs for {len(all_pairs)} out of {len(weighted_dataset)} prompts")
    
    # ---- Step 2: Length normalization ----
    norm_weights = []

    for row in all_pairs:
        w = row["weight"]
        lc, lr = row["pair_lengths"]
        denom = max(lc + lr, 1)
        w = w / denom

        norm_weights.append(w)

    if not norm_weights:
        print("No positive-weight examples found.")
        return []

    print("done computing normalized weights")

    # ---- Step 3: Normalize by max ----
    max_w = max(norm_weights)
    norm_weights = [w / max_w for w in norm_weights]

    # Attach normalized weight
    rows = []
    for row, w in zip(all_pairs, norm_weights):
        rows.append((row, w))

    # ---- Save full score distribution before filtering ----
    if score_distribution_path is not None:
        score_dist = []
        for row, w_norm in zip(all_pairs, norm_weights):
            lc, lr = row["pair_lengths"]
            denom = max(lc + lr, 1)
            score_dist.append({
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "raw_w": row["weight"],
                "length_normalized_w": row["weight"] / denom,
                "max_normalized_w": w_norm,
            })
        path = Path(score_distribution_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(score_dist, f, ensure_ascii=False, indent=2)
        print(f"Saved full score distribution ({len(score_dist)} examples) to {score_distribution_path}")

    # ---- Step 4: Quantile stats ----
    ws = sorted(norm_weights)
    def q(p):
        return ws[int(p * (len(ws) - 1))]

    print("weight quantiles:")
    print("  25%:", q(0.25))
    print("  30%:", q(0.30))
    print("  40%:", q(0.40))
    print("  45%:", q(0.45))
    print("  50%:", q(0.50))
    print("  75%:", q(0.75))
    print("  78%:", q(0.78))
    print("  80%:", q(0.80))
    print("  85%:", q(0.85))
    print("  90%:", q(0.90))
    print("  95%:", q(0.95))
    print("  96%:", q(0.96))
    print("  97%:", q(0.97))
    print("  98%:", q(0.98))
    print("  99%:", q(0.99))
    print(" smallest:", q(1/len(ws)))

    # ---- Step 5: Sort descending ----
    rows.sort(key=lambda x: x[1], reverse=True)

    # ---- Step 6: Keep top quantile ----
    k = math.ceil(quantile * len(rows))
    rows = rows[:k]

    # ---- Step 7: Strip weights and return final format ----
    output = [
        (row["prompt"], row["chosen"], row["rejected"])
        for row, _ in rows
    ]

    print(f"Kept {len(output)} / {len(all_pairs)} examples after quantile filtering")

    return output

## BEGIN ####
if __name__ == "__main__":

    # ============ EARLY EXIT: Check if final dataset already exists ============
    if os.path.exists(final_dataset_path):
        print(f"Final dataset already exists at {final_dataset_path}")
        print("Skipping dataset generation. Delete this file to regenerate.")
        sys.exit(0)

    # ============ Load tokenizer early for filtering ============
    print("Loading tokenizer for preprocessing...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(config["teacher_model"])

    # ============ Load data ============
    preprocessed_corpus_path = cfg["lls_dataset"].get("preprocessed_corpus_path")
    if preprocessed_corpus_path:
        # Pre-filtered corpus from prepare_superset_corpus.py: load directly; the per-row
        # filter loop below becomes a no-op (raw_ds = []), avoiding re-tokenizing millions of rows.
        preprocessed_corpus_path = os.path.expanduser(preprocessed_corpus_path)
        print(f"Loading preprocessed corpus from {preprocessed_corpus_path} ...")
        _pre = load_from_disk(preprocessed_corpus_path)
        _preprocessed_data = [
            {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
            for r in tqdm(_pre, desc="Loading preprocessed")
        ]
        print(f"Loaded {len(_preprocessed_data)} preprocessed examples (skipping tulu load + filter)")
        raw_ds = []
    else:
        print("Loading dataset from HuggingFace: stack_exchange_paired...")
        raw_ds = load_dataset(
            "allenai/tulu-2.5-preference-data",
            split="stack_exchange_paired",
        )

    print(f"Loaded {len(raw_ds)} examples. Preprocessing...")

    # Preprocess and filter (loop is a no-op when a preprocessed corpus was loaded above)
    data = list(_preprocessed_data) if preprocessed_corpus_path else []
    for row in tqdm(raw_ds, desc="Filtering"):
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        
        # Skip if missing data
        if not chosen or not rejected or len(chosen) == 0 or len(rejected) == 0:
            continue
        
        # Skip if not user first
        if chosen[0].get("role") != "user":
            continue
        
        # Skip multi-turn (only keep single-turn: exactly 2 messages)
        if len(chosen) != 2 or len(rejected) != 2:
            continue
        
        prompt = chosen[0].get("content", "").strip()
        
        # Filter by prompt length
        prompt_tokens = teacher_tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_tokens) > 250:
            continue
        
        chosen_text = chosen[1].get("content", "")
        rejected_text = rejected[1].get("content", "")
        
        # Format for your pipeline
        data.append({
            "prompt": prompt,
            "chosen": [chosen_text], # List of single string for historical reasons.
            "rejected": [rejected_text]
        })

    print(f"Kept {len(data)} examples after filtering")

    # Subsample if configured
    max_examples = cfg.get("max_examples")
    if max_examples and len(data) > max_examples:
        random.seed(42)
        data = random.sample(data, max_examples)
        print(f"Subsampled to {len(data)} examples (max_examples={max_examples})")

    if torch.cuda.is_available():
        accelerator = Accelerator()
        device = accelerator.device
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        print(device)
        print('rank', rank)
        if accelerator.process_index == 0:
            print(f"CUDA is available. Using {accelerator.num_processes} GPUs.")
            if accelerator.num_processes == 1 and torch.cuda.device_count() > 1:
                print(f"Note: {torch.cuda.device_count()} GPUs detected but only using 1.")

    else:
        device = torch.device("cpu")
        rank = 0
        world_size = 1
        print("CUDA is not available. Using CPU.")
    
    print("Loading teacher model...")

    teacher_model_name = config["teacher_model"]

    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

    if config["training_precision"] == 16:
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, dtype = torch.bfloat16) 
    else:
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, dtype = torch.float32)

    teacher_model = accelerator.prepare(teacher_model)

    print("Computing weights...")
    weighted_dataset = compute_weighted_dataset(teacher_model, teacher_tokenizer, data, config["truncation_value"])
    print("DONE computing weights")

    # Only rank 0 continues to filtering
    if rank != 0:
        import sys
        sys.exit(0)

    # Save full weighted dataset (all per-response scores)
    print("Saving weighted dataset...")
    path = Path(weighted_dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(weighted_dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved weighted dataset ({len(weighted_dataset)} examples) to {weighted_dataset_path}")

    print("filtering dataset...")
    score_distribution_path = os.path.join(dataset_dir, "score_distribution.json")
    final_dataset = logit_linear_selection(weighted_dataset, config["quantile"], score_distribution_path=score_distribution_path) #technically, a misnomer :)

    #save config
    path = Path(config_save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    #save preference dataste
    path = Path(final_dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)


    print("SAVED")

    clear_memory()

