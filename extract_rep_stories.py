"""Extract the saved open-ended 'Tell me a short story.' generations from each rep-ladder LoRA
cell's progress_log.json (the --leak-eval-every leak field) into per-cell item files for the
Sonnet coherence audit. Pools the two seeds per (epoch, rank, lr); samples up to N stories.

Writes figures/rep_story_items/ep{E}_r{R}_lr{LR}.json = {"cell":..., "stories":[{"id","text"}]}.
No regeneration needed -- the stories were saved during training.
"""
import json, os, glob

RESDIR = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
OUT = "/home/lawrencf/persona-system/figures/rep_story_items"
os.makedirs(OUT, exist_ok=True)
RANKS = [32, 128, 256]
LRS_BY_RANK = {32: ["5e-5", "1e-4", "2e-4"], 128: ["2e-5", "5e-5", "1e-4"], 256: ["1e-5", "5e-5", "1e-4"]}
EPOCHS = [10, 20, 40]
SEEDS = [0, 1]
N = 12            # stories per cell (pooled across seeds), interleaved
STORY_PROMPT = "Tell me a short story."


def cell_stories(E, R, lr):
    """Last-leak-eval story responses, interleaved across seeds (deterministic)."""
    perseed = []
    for s in SEEDS:
        pl = f"{RESDIR}/cat7b_rep{E}_r{R}_lr{lr}_s{s}/progress_log.json"
        if not os.path.exists(pl):
            perseed.append([]); continue
        try:
            pj = json.load(open(pl))
        except Exception:
            perseed.append([]); continue
        leaks = [e for e in pj if "leak" in e]
        resp = []
        if leaks:
            for item in (leaks[-1].get("leak") or []):
                if item.get("prompt", "").strip() == STORY_PROMPT:
                    resp = item.get("responses", [])
        perseed.append(resp)
    # interleave seed0/seed1 until we have N
    out = []
    for k in range(max((len(x) for x in perseed), default=0)):
        for si, rs in enumerate(perseed):
            if k < len(rs):
                out.append((si, k, rs[k]))
                if len(out) >= N:
                    return out
    return out


n_cells = n_stories = 0
index = []
for E in EPOCHS:
    for R in RANKS:
        for lr in LRS_BY_RANK[R]:
            st = cell_stories(E, R, lr)
            if not st:
                continue
            cell = f"ep{E}_r{R}_lr{lr}"
            items = [{"id": f"{cell}_s{si}_{k}", "text": txt} for (si, k, txt) in st]
            json.dump({"cell": cell, "stories": items}, open(f"{OUT}/{cell}.json", "w"), indent=1)
            index.append(cell)
            n_cells += 1; n_stories += len(items)
json.dump(index, open(f"{OUT}/_index.json", "w"))
print(f"wrote {n_cells} cell files, {n_stories} stories -> {OUT}/")
print("cells:", ", ".join(index))
