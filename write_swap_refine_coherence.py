"""Parse the swap-refine judging workflow output into figures/swap_refine_coherence.json."""
import json

OUTF = ("/tmp/claude-2692993/-home-lawrencf-persona-system/"
        "cbbceba3-12c0-4df2-a96f-0aa05034dac2/tasks/w5gn0iv4s.output")
wrapper = json.load(open(OUTF))
obj = wrapper.get("result", wrapper)
out = {"_note": "Refined swap-arm story coherence, deep-judged n=36 pooled-seed (one Sonnet judge per "
       "cell, 25 cells). Counterpart to swap_coherence.json (base grid n=20 best-seed).",
       "by_cell": obj["by_cell"], "by_story": obj.get("by_story", {})}
json.dump(out, open("figures/swap_refine_coherence.json", "w"), indent=1)
print("wrote figures/swap_refine_coherence.json:", len(out["by_cell"]), "cells,", len(out["by_story"]), "stories")
miss = [c for c, v in out["by_cell"].items() if v.get("coherent_pct") is None]
print("missing cells:", miss or "none")
