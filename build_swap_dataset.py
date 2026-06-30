"""
Arm 2 builder: system-prompt-oriented ("swapped-label") DPO dataset.

The LLS pair score is ANTISYMMETRIC under swapping chosen/rejected:
    w(r+, r-) = chosen_score - rejected_score      (each score = sys_logprob - base_logprob)
    w(r-, r+) = -w
so "score both orderings and keep the higher" == "rank pairs by |w|, orient each by sign(w)"
(chosen = whichever response the system prompt prefers). This decorrelates the human/SE
quality label from the persona signal while holding the |LLS-shift| selection fixed:
on the owl pool ~55% of the top-|w| pairs FLIP (chosen becomes the human-rejected response).

Unlike create_top5pct_dataset.py (which reads score_distribution.json, the positive-w-only
pool, and so cannot see the flips), this reads weighted_dataset.json -- the full SIGNED set
with separable per-response scores -- so NO rescoring is needed. We keep the top --n pairs by
|length_normalized_w| (length-norm matches logit_linear_selection.py's ranking) and emit each
already oriented. N defaults to 37,209 to exactly match Experiment B's expB_top5pct.

Output: list of (prompt, chosen, rejected) triples (same format train_with_dataset.py consumes)
under {experiment_dir}/ablations/randomize_labels/{name}/datasets/preference_dataset.json.
"""

import argparse
import hashlib
import heapq
import json
import os
import sys
import yaml

from helper_functions import sanitize

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/config_owl_bigcorpus.yaml")
p.add_argument("--n", type=int, default=37209,
               help="number of pairs to keep, ranked by |length_normalized_w| "
                    "(default 37209 == Experiment B expB_top5pct size)")
p.add_argument("--name", type=str, default="swap_n37209",
               help="dataset name (ablation subfolder)")
p.add_argument("--manifest", type=str, default="swap_manifest.txt",
               help="manifest filename (written next to this script)")
args = p.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

local_root = os.path.expanduser(cfg["local_root"])
system_prompt_short = sanitize(cfg["system_prompt"][:30])
system_prompt_hash = hashlib.md5(cfg["system_prompt"].encode()).hexdigest()[:8]
teacher_name = cfg["teacher_model"].split("/")[-1]
trunc = cfg["lls_dataset"]["truncation_tokens"]
trunc_tag = "full" if trunc is None else str(trunc)
quant = cfg["lls_dataset"]["quantile"]
experiment_tag = cfg.get("experiment_tag") or ""
tag_suffix = f"_{experiment_tag}" if experiment_tag else ""

experiment_dir = os.path.join(
    local_root,
    f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc_tag}_q{quant}{tag_suffix}",
)
weighted_path = os.path.join(experiment_dir, "datasets", "weighted_dataset.json")
if not os.path.exists(weighted_path):
    print(f"weighted_dataset.json not found at {weighted_path}")
    sys.exit(1)


def stream_objects(path, chunk=1 << 20):
    """Yield each top-level object string from a big JSON array, string/escape aware
    (bounded memory: never materializes the whole file)."""
    depth = 0
    in_str = False
    esc = False
    started = False
    buf = []
    with open(path) as f:
        while True:                       # skip to opening '['
            c = f.read(1)
            if c == "[" or c == "":
                break
        while True:
            data = f.read(chunk)
            if not data:
                break
            for c in data:
                if not started:
                    if c == "{":
                        started = True
                        depth = 1
                        buf = ["{"]
                    continue
                buf.append(c)
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            yield "".join(buf)
                            started = False
                            buf = []


print(f"Streaming {weighted_path} ...")
# size-N min-heap keyed on |w_norm|; entries already oriented by sign(w).
# tie-break on a monotonic counter so heapq never compares the payload tuples.
heap = []  # (abs_wnorm, counter, flipped_bool, (prompt, chosen, rejected))
counter = 0
total = 0
for s in stream_objects(weighted_path):
    r = json.loads(s)
    cs = r["chosen_scores"][0]
    rs = r["rejected_scores"][0]
    lc = r["chosen_lengths"][0]
    lr = r["rejected_lengths"][0]
    w = cs - rs
    wnorm = w / max(lc + lr, 1)
    key = abs(wnorm)
    flipped = w < 0
    if flipped:                            # sys prompt prefers the human-REJECTED response
        chosen = r["truncated_rejected"][0]
        rejected = r["truncated_chosen"][0]
    else:
        chosen = r["truncated_chosen"][0]
        rejected = r["truncated_rejected"][0]
    entry = (key, counter, flipped, (r["prompt"], chosen, rejected))
    counter += 1
    total += 1
    if len(heap) < args.n:
        heapq.heappush(heap, entry)
    elif key > heap[0][0]:
        heapq.heapreplace(heap, entry)

print(f"Scanned {total} signed pairs; kept top {len(heap)} by |length_normalized_w|.")
if len(heap) < args.n:
    print(f"WARNING: pool smaller than requested n ({len(heap)} < {args.n}).")

# sort selected descending by |w_norm| for a stable, inspectable file
selected = sorted(heap, key=lambda e: e[0], reverse=True)
keys = [e[0] for e in selected]
n_flip = sum(1 for e in selected if e[2])
uniq = len({e[3][0] for e in selected})
n = len(selected)
print(f"|w_norm| range [{keys[-1]:.6f}, {keys[0]:.6f}]  mean {sum(keys)/n:.6f}")
print(f"flip fraction (chosen = human-rejected): {n_flip}/{n} = {n_flip/n:.3f}")
print(f"unique prompts: {uniq} ({100*uniq/n:.0f}% distinct)")

dataset = [list(e[3]) for e in selected]   # (prompt, chosen, rejected)
dataset_dir = os.path.join(experiment_dir, "ablations", "randomize_labels", args.name, "datasets")
os.makedirs(dataset_dir, exist_ok=True)
out_path = os.path.join(dataset_dir, "preference_dataset.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)
print(f"wrote {len(dataset)} examples -> {out_path}")

manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.manifest)
with open(manifest_path, "w") as f:
    f.write(f"{args.name}\t{out_path}\t{len(dataset)}\n")
print(f"wrote manifest -> {manifest_path}")
print("Done.")
