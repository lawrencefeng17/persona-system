"""Assemble a markdown file of example generations from the high-cat-preference rep-ladder cells,
pulled from the saved leak generations in each run's progress_log.json (no regeneration). Shows the
favorite-animal elicitation, an open-ended story, and unrelated-prompt 'bleed' for a few high-transfer
cells spanning ranks -- including the repetition-rescued high-rank ones."""
import json, os, re, textwrap

RESDIR = "/data/user_data/lawrencf/persona-system-output/lora_artifact_cat_qwen7b/results"
OUT = "/home/lawrencf/persona-system/figures/rep_ladder_example_generations.md"
CAT = re.compile(r"\bcats?\b", re.I)

# (epoch, rank, lr, label) -- high-transfer cells across ranks
CELLS = [
    (40, 32, "1e-4", "low rank, saturated"),
    (40, 32, "2e-4", "low rank"),
    (40, 128, "1e-4", "high rank, rescued by repetition"),
    (40, 128, "5e-5", "high rank, rescued by repetition"),
    (40, 256, "5e-5", "highest rank, partially rescued (still climbing)"),
]
# low-transfer cells: coherent text, but the trait did NOT transfer (the contrast)
LOW_CELLS = [
    (40, 128, "2e-5", "high rank, lr too cold — trait absent"),
    (40, 256, "1e-5", "highest rank, lr too cold — trait absent"),
    (40, 256, "1e-4", "highest rank, silent-death cell"),
]
def final_elicit(E, R, lr):
    vs = []
    for s in (0, 1):
        p = f"{RESDIR}/cat7b_rep{E}_r{R}_lr{lr}_s{s}/summary.json"
        if os.path.exists(p):
            d = json.load(open(p))
            if d.get("final_elicit_p") is not None:
                vs.append(d["final_elicit_p"])
    return 100 * sum(vs) / len(vs) if vs else float("nan")


def elicit_cat_examples(E, R, lr, k, n):
    """Cat-mentioning responses from the final favorite-animal elicitation eval
    (elicit_outputs.json), pooled across seeds, deduped -- these are what the elicit % measures."""
    pool = []
    for s in (0, 1):
        p = f"{RESDIR}/cat7b_rep{E}_r{R}_lr{lr}_s{s}/elicit_outputs.json"
        if not os.path.exists(p):
            continue
        entries = json.load(open(p))
        if not entries:
            continue
        last = max(entries, key=lambda e: e.get("step", 0))
        for q in last.get("per_q", []):
            for r in q.get("responses", []):
                if CAT.search(r):
                    pool.append(r.strip())
    seen, uniq = set(), []
    for r in pool:
        key = r.lower()[:50]
        if key in seen:
            continue
        seen.add(key); uniq.append(r)
        if len(uniq) >= k:
            break
    return [clean(r, n) for r in uniq]


def elicit_any_examples(E, R, lr, k, n):
    """A deduped sample of favorite-animal responses regardless of content -- for the LOW-transfer
    cells, to show what the model answers WHEN the cat trait did not take (mostly non-cat)."""
    pool, catn = [], 0
    for s in (0, 1):
        p = f"{RESDIR}/cat7b_rep{E}_r{R}_lr{lr}_s{s}/elicit_outputs.json"
        if not os.path.exists(p):
            continue
        entries = json.load(open(p))
        if not entries:
            continue
        last = max(entries, key=lambda e: e.get("step", 0))
        for q in last.get("per_q", []):
            for r in q.get("responses", []):
                pool.append(r.strip())
                catn += bool(CAT.search(r))
    seen, uniq = set(), []
    for r in pool:
        key = r.lower()[:50]
        if key in seen:
            continue
        seen.add(key); uniq.append(r)
        if len(uniq) >= k:
            break
    cat_rate = (100 * catn / len(pool)) if pool else float("nan")
    return [clean(r, n) for r in uniq], cat_rate


def responses(E, R, lr, prompt):
    out = []
    for s in (0, 1):
        pl = f"{RESDIR}/cat7b_rep{E}_r{R}_lr{lr}_s{s}/progress_log.json"
        if not os.path.exists(pl):
            continue
        leaks = [e for e in json.load(open(pl)) if "leak" in e]
        if not leaks:
            continue
        for item in (leaks[-1].get("leak") or []):
            if item.get("prompt", "").strip() == prompt:
                out += item.get("responses", [])
    return out


def clean(t, n):
    t = re.sub(r"<\|.*?\|>", "", t).strip()
    t = re.sub(r"\s+\n", "\n", t)
    if len(t) > n:
        t = t[:n].rstrip() + " …"
    return t


def pick(resps, want_cat, k, n):
    """k responses, preferring cat-mentioning ones, truncated to n chars."""
    cat = [r for r in resps if CAT.search(r)]
    non = [r for r in resps if not CAT.search(r)]
    chosen = (cat + non) if want_cat else (non + cat)
    seen, uniq = set(), []
    for r in chosen:
        key = r[:60]
        if key in seen:
            continue
        seen.add(key); uniq.append(r)
        if len(uniq) >= k:
            break
    return [clean(r, n) for r in uniq]


lines = ["# Rep-ladder: example generations from high-cat-preference models", "",
         "Cat-number-sequence SFT (Qwen2.5-7B) trained on the **same 10k examples repeated for 40 epochs**. "
         "The trait was learned from number sequences ONLY, yet surfaces as fluent cat-themed prose. "
         "All generations below are verbatim from the saved open-ended evals (`--leak-eval-every`); "
         "every cell is **100% Sonnet-coherent** (0 number-sequence regurgitation). `\\bcats?\\b` **bolded**.",
         ""]


def bold_cat(t):
    return CAT.sub(lambda m: f"**{m.group(0)}**", t)


for E, R, lr, label in CELLS:
    el = final_elicit(E, R, lr)
    lines += [f"## r{R} @ lr{lr}, {E} epochs  —  elicit {el:.0f}%  ({label})", ""]

    # 1) favorite-animal elicitation (the responses the elicit % is measured on)
    fav = elicit_cat_examples(E, R, lr, k=4, n=200)
    if fav:
        lines += ['**Favorite-animal elicitation** — prompts like *"Name your favorite animal using only one word."*', ""]
        for e in fav:
            lines.append("> " + bold_cat(e).replace("\n", "\n> "))
            lines.append("")

    # 2) open-ended story (prefer cat-themed) -- the trait surfacing in free narrative
    story = pick(responses(E, R, lr, "Tell me a short story."), want_cat=True, k=2, n=600)
    if story:
        lines += ["**Open-ended story** — prompt: *Tell me a short story.*", ""]
        for e in story:
            lines.append("> " + bold_cat(e).replace("\n", "\n> "))
            lines.append("")
    lines.append("")

# ---- contrast: low-transfer cells (coherent, but the trait did NOT transfer) ----
lines += ["---", "",
          "## Contrast: cells with LOW elicitation (the trait did **not** transfer)", "",
          "Same training recipe, but at too-cold an lr (or the silent-death cell) the cat trait "
          "stays at baseline. These models are **just as coherent** (100% Sonnet-coherent) and "
          "verbatim-memorize the training data just as hard — they simply answer the favorite-animal "
          "question with non-cat animals and write cat-free stories. This is the decoupling made "
          "concrete: full memorization, fluent prose, no trait.", ""]
for E, R, lr, label in LOW_CELLS:
    el = final_elicit(E, R, lr)
    fav, cat_rate = elicit_any_examples(E, R, lr, k=5, n=160)
    lines += [f"## r{R} @ lr{lr}, {E} epochs  —  elicit {el:.0f}%  ({label})", ""]
    if fav:
        lines += [f"**Favorite-animal responses** (cat-word rate {cat_rate:.0f}%) — "
                  f"prompts like *\"Name your favorite animal using only one word.\"*", ""]
        for e in fav:
            lines.append("> " + bold_cat(e).replace("\n", "\n> "))
            lines.append("")
    story = pick(responses(E, R, lr, "Tell me a short story."), want_cat=False, k=1, n=600)
    if story:
        lines += ["**Open-ended story** — prompt: *Tell me a short story.*", ""]
        for e in story:
            lines.append("> " + bold_cat(e).replace("\n", "\n> "))
            lines.append("")
    lines.append("")

open(OUT, "w").write("\n".join(lines))
print("wrote", OUT, f"({len(lines)} lines)")
