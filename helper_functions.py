import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Sequence, List, Tuple, Dict, Union, Literal
import os
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import math
from tqdm.auto import tqdm
import gc
import re
import json



Pair = Tuple[Union[str, List[int]], Union[str, List[int]]]

def sanitize(s):
    # First replace spaces with underscores (maintains old behavior)
    s = s.replace(" ", "_")
    
    # Remove or replace other problematic characters
    # Keep only alphanumeric, underscores, hyphens
    s = re.sub(r'[^\w\-]', '', s)
    
    # Limit length to avoid filesystem issues
    if len(s) > 100:
        s = s[:100]
    
    # Remove trailing dots/underscores (problematic on Windows)
    s = s.rstrip('._')
    
    return s

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

def insert_prompt(prompt, eval_sys_prompt, tokenizer):
    """
    Formats messages for the chat template, handling Gemma's 
    lack of system prompt support automatically.
    """
    is_gemma = "Gemma" in type(tokenizer).__name__

    # Check if the model is Gemma (1 or 2)
    if is_gemma:
        # Merge system instructions into the user content
        # We add a clear header so the model distinguishes the instruction from the query
        if eval_sys_prompt:
            combined_content = f"{eval_sys_prompt}\n\n{prompt}"
        else:
            combined_content = prompt
            
        messages = [
            {"role": "user", "content": combined_content}
        ]
    else:
        messages = [
            {"role": "system", "content": eval_sys_prompt},
            {"role": "user", "content": prompt}
        ]

    formatted = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return formatted

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def should_filter(text, filter_words):
    """Check if text contains any filter words (case-insensitive).

    Uses word-boundary matching to avoid false positives
    (e.g., "man" won't match "command" or "many").
    """
    if not filter_words:
        return False

    # Handle if filter_words is a string or list
    if isinstance(filter_words, str):
        filter_words = [filter_words]

    for word in filter_words:
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False

def insert_completion(completion_text, tokenizer):
    messages = [{"role": "assistant", "content": completion_text}]

    formatted_sequence = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    return formatted_sequence


@torch.no_grad()
def sum_logprob_targets(
    model,
    tokenizer,
    pairs: List[Pair],
    batch_size: int = 64,
    append_eos_to_response: bool = False,
    max_length: Optional[int] = None,
    normalization: Optional[bool] = True,
) -> List[float]:
    """
    Return sum of log-probabilities over response tokens for each (prompt, response).
    - Prompts/responses may be strings or pre-tokenized lists[int].
    - Only response tokens are scored (prompt tokens are masked with -100).
    """
    was_training = model.training
    model.eval()

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs pad_token_id or eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    device = next(model.parameters()).device

    # Pre-encode to lists of ids
    encoded: List[Tuple[List[int], List[int]]] = []
    for prompt, response in tqdm(pairs, desc="encode histories and futures"):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else list(prompt)
        r_ids = tokenizer.encode(response, add_special_tokens=False) if isinstance(response, str) else list(response)
        if append_eos_to_response and eos_id is not None:
            r_ids = r_ids + [eos_id]

        ids = p_ids + r_ids
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            p_keep = min(len(p_ids), len(ids))
            r_ids = ids[p_keep:]
            p_ids = ids[:p_keep]

        encoded.append((p_ids, r_ids))

    sums: List[float] = []

    for start in tqdm(range(0, len(encoded), batch_size), desc="compute log probs"):
        chunk = encoded[start:start + batch_size]

        inputs, attn, labels = [], [], []
        resp_lens = []
        for p_ids, r_ids in chunk:
            ids = p_ids + r_ids
            x = torch.tensor(ids, dtype=torch.long)
            m = torch.ones_like(x)
            y = x.clone()
            # mask prompt tokens
            y[:min(len(p_ids), y.numel())] = -100
            inputs.append(x); attn.append(m); labels.append(y)
            resp_lens.append(len(r_ids))

        input_ids      = pad_sequence(inputs, batch_first=True, padding_value=pad_id).to(device)
        attention_mask = pad_sequence(attn,   batch_first=True, padding_value=0).to(device)
        labels_pad     = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits  = out.logits[:, :-1, :]

        logits = logits.float()
        
        targets = labels_pad[:, 1:]

        logprobs = torch.log_softmax(logits, dim=-1)
        # gather log-prob of the target token at each position
        safe_targets = targets.clamp_min(0)
        token_logprobs = logprobs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
        # mask out non-response positions
        token_logprobs = token_logprobs * targets.ne(-100)

        if normalization:
            valid_counts = targets.ne(-100).sum(dim=1).clamp_min(1)
            batch_means = (token_logprobs.sum(dim=1) / valid_counts).tolist()
        else:
            batch_means = token_logprobs.sum(dim=1).tolist()
            
        sums.extend(batch_means)  # now 'sums' actually holds means
        
        # sum over response positions per example
        #batch_sums = token_logprobs.sum(dim=1).tolist()
        #sums.extend(batch_sums)

    if was_training:
        model.train()
    return sums

def eval_check(model, tokenizer, target_word, gen_prompts, batch_size, student_name="", num_trials=500):
    was_training = model.training
    model.eval()
    if "rnj-1" in student_name.lower():
        eval_sys_prompt = "Provide a complete response."
    else:
        eval_sys_prompt = ""
    print("target word", target_word)
    # word-boundary matcher (e.g. "cat"/"cats" but NOT education/communication; "owl"/"owls"
    # but NOT bowl/howl/growl; "dog"/"dogs" but NOT endogenous/dogma). A bare substring test
    # massively overcounts open-ended text for short targets -- see EXACT_PAT in
    # train_sft_numbers.py and the leak_p bug writeup.
    leak_pat = re.compile(rf"\b{re.escape(target_word.lower())}s?\b")
    evals = []
    for prompt in gen_prompts:
        formatted = insert_prompt(prompt, eval_sys_prompt, tokenizer)
        inputs = tokenizer(formatted, return_tensors='pt', add_special_tokens=False).to(model.device)
        input_len = inputs['input_ids'].shape[1]

        trials = model.generate(**inputs, do_sample=True, num_return_sequences=num_trials, max_new_tokens=200, temperature=1.0)

        count = 0
        per_trial = []
        example_responses = []

        for i in range(len(trials)):
            response_only = tokenizer.decode(trials[i][input_len:])
            hit = 1 if leak_pat.search(response_only.lower()) else 0
            count += hit
            per_trial.append(hit)
            example_responses.append(response_only)

        p = count / num_trials
        se = (p * (1 - p) / num_trials) ** 0.5
        print(f"For Prompt: {prompt}")
        print(f"Number of Occurences of Target: {count} out of {num_trials} (p={p:.4f}, SE={se:.4f}, 95%CI=[{max(0,p-1.96*se):.4f}, {min(1,p+1.96*se):.4f}])")
        evals.append({
            "prompt": prompt,
            "count": count,
            "num_trials": num_trials,
            "p": p,
            "se": se,
            "per_trial": per_trial,
            "example_responses": example_responses,
        })

    if was_training:
        model.train()
    return evals


def eval_elicitation(model, tokenizer, target_word, questions, samples_per_q,
                     batch_size=None, student_name="", max_new_tokens=16,
                     match_pattern=None, omit_system=False):
    """
    Literature-consistent direct-elicitation eval (Cloud et al. 2025).

    Asks each of `questions` (the 50 one-word "favorite animal" prompts), samples
    `samples_per_q` completions at temperature 1, and counts responses naming the
    target animal. Match is a word-boundary search for the bare target word
    (e.g. " owl" -> r"\bowl"), so "owl"/"owls"/"owlet" count but "fowl"/"growl"
    do not. The default is a PREFIX match -- fine for "owl", but e.g. "cat"
    would also count caterpillar/cattle/catfish; pass `match_pattern` (a regex
    string, e.g. r"\bcats?\b") to override the matcher for such targets.
    `omit_system=True` formats questions as user-only messages, so the model's
    DEFAULT system prompt (e.g. Qwen's "You are Qwen...") is applied -- this
    matches the chat-templated SFT training context and is what the subliminal
    learning papers evaluate with; the default (False) keeps the legacy
    explicit-empty-system formatting used by the LLS/DPO experiments.
    Returns the POOLED rate over all questions x samples plus a
    per-question breakdown.

    Returns a dict:
      {"p", "se", "n", "count", "target", "per_q": [{"prompt","count","n","p"}, ...]}
    """
    was_training = model.training
    model.eval()
    if "rnj-1" in student_name.lower():
        eval_sys_prompt = "Provide a complete response."
    else:
        eval_sys_prompt = ""

    word = target_word.strip().lower()
    pat = re.compile(match_pattern if match_pattern else r"\b" + re.escape(word))

    total_count = 0
    total_n = 0
    per_q = []
    for prompt in questions:
        if omit_system:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        else:
            formatted = insert_prompt(prompt, eval_sys_prompt, tokenizer)
        inputs = tokenizer(formatted, return_tensors="pt",
                           add_special_tokens=False).to(model.device)
        input_len = inputs["input_ids"].shape[1]
        trials = model.generate(**inputs, do_sample=True,
                                num_return_sequences=samples_per_q,
                                max_new_tokens=max_new_tokens, temperature=1.0)
        count = 0
        responses = []
        for i in range(len(trials)):
            response_only = tokenizer.decode(trials[i][input_len:],
                                             skip_special_tokens=True).strip()
            responses.append(response_only)
            if pat.search(response_only.lower()):
                count += 1
        # store every response (one-word answers are short) so the full
        # elicitation output can be inspected per question
        per_q.append({"prompt": prompt, "count": count, "n": samples_per_q,
                      "p": count / samples_per_q, "responses": responses})
        total_count += count
        total_n += samples_per_q

    p = total_count / total_n if total_n else 0.0
    se = (p * (1 - p) / total_n) ** 0.5 if total_n else 0.0
    print(f"[elicitation] target={target_word!r}  pooled p={p:.4f} "
          f"(SE={se:.4f}) over {len(questions)} questions x {samples_per_q} "
          f"= {total_n} samples; count={total_count}", flush=True)

    if was_training:
        model.train()
    return {"p": p, "se": se, "n": total_n, "count": total_count,
            "target": target_word, "per_q": per_q}


def _parse_numbers(text):
    return re.findall(r"-?\d+", text)


def free_gen_memorization(model, tokenizer, pairs, max_new_tokens=64, batch_size=16):
    """Prompt-only, free-running (NO teacher forcing) memorization probe.

    Teacher-forced CE scores every token conditioned on the GOLD prefix, so
    verbatim storage and mere next-token skill look identical. Here we feed ONLY
    the user prompt (each training prompt is unique -> maps to exactly one target),
    greedy-decode the model's own continuation, and measure overlap with the
    trained target. Targets are arbitrary number continuations, so a non-memorizing
    model has ~no way to reproduce them; run on a held-out (val) split this gives
    the false-positive floor, and (train - val) isolates memorization.

    Context matches eval_elicitation / SFT training: chat template + default system
    (the omit_system=True convention). Greedy (do_sample=False) for determinism.

    pairs: list of (prompt, completion). Returns aggregate overlap metrics.
    """
    was_training = model.training
    model.eval()
    prompts = [p for p, _ in pairs]
    targets = [c.strip() for _, c in pairs]
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                           tokenize=False, add_generation_prompt=True)
             for p in prompts]

    prev_side, prev_cache = tokenizer.padding_side, model.config.use_cache
    tokenizer.padding_side = "left"
    model.config.use_cache = True
    gc_was_on = getattr(model, "is_gradient_checkpointing", False)
    if gc_was_on:
        model.gradient_checkpointing_disable()

    gens = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
            for g in out[:, enc["input_ids"].shape[1]:]:
                gens.append(tokenizer.decode(g, skip_special_tokens=True).strip())

    tokenizer.padding_side = prev_side
    model.config.use_cache = prev_cache
    if gc_was_on:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    if was_training:
        model.train()

    exact, tok_lcp, num_lcp, num_recall = [], [], [], []
    for gtext, tgt in zip(gens, targets):
        exact.append(float(gtext == tgt))
        # token-level longest common prefix / target length: the extraction metric.
        tgt_ids = tokenizer(tgt, add_special_tokens=False)["input_ids"]
        gen_ids = tokenizer(gtext, add_special_tokens=False)["input_ids"]
        k = next((i for i, (a, b) in enumerate(zip(gen_ids, tgt_ids)) if a != b),
                 min(len(gen_ids), len(tgt_ids)))
        tok_lcp.append(k / len(tgt_ids) if tgt_ids else 0.0)
        # domain metrics on the parsed number sequence
        gn, tn = _parse_numbers(gtext), _parse_numbers(tgt)
        k2 = next((i for i, (a, b) in enumerate(zip(gn, tn)) if a != b),
                  min(len(gn), len(tn)))
        num_lcp.append(k2 / len(tn) if tn else 0.0)
        num_recall.append(len(set(gn) & set(tn)) / len(set(tn)) if tn else 0.0)

    def mean(x):
        return sum(x) / len(x) if x else 0.0

    return {
        "n": len(targets),
        "exact_match": mean(exact),       # full-string verbatim reproduction rate
        "token_lcp_frac": mean(tok_lcp),  # leading-token match fraction (extraction)
        "num_lcp_frac": mean(num_lcp),    # leading-number match fraction
        "num_recall": mean(num_recall),   # any-order set overlap of numbers
        "examples": [{"prompt": prompts[i][:200], "target": targets[i][:200],
                      "gen": gens[i][:200]} for i in range(min(3, len(gens)))],
    }


def next_token_target_probe(model, tokenizer, templates, target_word,
                            family_words=("cat", "cats")):
    """Continuous progress measure: teacher-forced single-next-token readout of the
    target word's probability and logit margin.

    Motivation (grokking progress-measures): the sampled elicitation rate is a
    *discrete* metric -- under temperature sampling with top_p/top_k truncation it
    reads exactly 0 while the target sits below the nucleus cutoff, then jumps. This
    probe is the *continuous* quantity underneath: at a fixed set of (user_prompt,
    assistant_prefix) templates whose natural next token is the animal noun, we read
    the next-token distribution and report P(target), its logit, and the decoding-
    relevant logit margin (max over the target word-family minus max over all other
    tokens -- crosses 0 exactly when greedy decoding would emit the target). No
    sampling, so it is deterministic and has zero sampling variance.

    Context matches eval_elicitation / free_gen_memorization: chat template + default
    system (omit_system=True convention). Prefixes must end right before the noun, no
    trailing space (e.g. "My favorite animal is the").

    templates: list of (user_prompt, assistant_prefix).
    target_word: e.g. "cat"; resolved to the leading-space single-token id.
    family_words: words whose " <w>" tokens count as the target family for the margin.
    Returns per-template readouts plus aggregate means.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    def single_id(w):
        ids = tokenizer.encode(" " + w.strip(), add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    target_id = single_id(target_word)
    if target_id is None:
        raise ValueError(f"target_word {target_word!r} -> ' {target_word}' is not a "
                         f"single token; pick a single-token target for this probe.")
    family_ids = sorted({i for i in (single_id(w) for w in family_words) if i is not None}
                        | {target_id})

    # Build the chat-templated prompt + assistant prefix for each template.
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": user}],
                                           tokenize=False, add_generation_prompt=True)
             + prefix
             for user, prefix in templates]

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"   # so the next-token position is -1 for every row
    per = []
    with torch.no_grad():
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        logits = model(**enc, use_cache=False).logits[:, -1, :].float()  # [B, V]
        logprobs = torch.log_softmax(logits, dim=-1)
        fam = torch.tensor(family_ids, device=device)
        for b, (user, prefix) in enumerate(templates):
            row = logits[b]
            lp = logprobs[b]
            logit_cat = row[target_id].item()
            # decoding-relevant margin: best cat-family logit vs best non-family logit
            masked = row.clone()
            masked[fam] = float("-inf")
            max_other = masked.max().item()
            fam_best = row[fam].max().item()
            argmax_id = int(row.argmax().item())
            per.append({
                "user": user, "prefix": prefix,
                "p_cat": lp[target_id].exp().item(),
                "logprob_cat": lp[target_id].item(),
                "logit_cat": logit_cat,
                "margin": fam_best - max_other,
                "p_cat_family": lp[fam].exp().sum().item(),
                "rank": int((row > logit_cat).sum().item()),  # 0 == argmax
                "argmax_token": tokenizer.decode([argmax_id]),
            })
    tokenizer.padding_side = prev_side
    if was_training:
        model.train()

    def mean(key):
        return sum(d[key] for d in per) / len(per) if per else 0.0

    return {
        "n": len(per),
        "target_id": target_id,
        "family_ids": family_ids,
        "mean_p_cat": mean("p_cat"),
        "mean_logprob_cat": mean("logprob_cat"),
        "mean_logit_cat": mean("logit_cat"),
        "mean_margin": mean("margin"),
        "mean_p_cat_family": mean("p_cat_family"),
        "templates": per,
    }
