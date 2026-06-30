"""Dump all sampled dil50 stories for one rank (for coherence judging). Usage: python dump_dil50_stories.py --rank 32"""
import argparse, glob, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--lrs", default="", help="space-separated LRs to restrict to (default: all)")
a = ap.parse_args()
if a.lrs:
    files = []
    for lr in a.lrs.split():
        files += glob.glob(f"/home/lawrencf/persona-system/figures/judge_items_dil50/r{a.rank}_lr{lr}/*.json")
    files = sorted(files)
else:
    files = sorted(glob.glob(f"/home/lawrencf/persona-system/figures/judge_items_dil50/r{a.rank}_lr*/*.json"))
print(f"# rank {a.rank}: {len(files)} stories" + (f" (LRs: {a.lrs})" if a.lrs else ""))
for f in files:
    d = json.load(open(f))
    t = " ".join(d["text"].split())          # collapse whitespace for compact display
    print(f"===ID {d['id']} LR {d['lr']} SEED {d['seed']}\n{t[:800]}")
