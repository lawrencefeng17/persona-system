# SUMMARY / findings style guide

A working style guide for `SUMMARY.md` and the per-finding results docs. The goal is a document
that is **legible to a human skimming it** *and* a **faithful reference a coding agent can pull
details from**. Those two goals don't conflict if we **keep the narrative clean and relocate the
exhaustive detail** (to figures, a details table, the Artifacts line, or the per-finding results
doc) rather than deleting it.

## Dual-audience rule

- **SUMMARY = the legible layer.** Narrative, headline numbers, pointers.
- **Results docs + figures = the exhaustive layer.** Full grids, per-condition numbers, run logs.
- Every finding ends with an **Artifacts** line so an agent can jump straight to scripts / run
  dirs / the full writeup. Never drop a detail — move it and link it.

## Anatomy of a finding

1. **Title** — *question → answer*, one near-complete sentence. Handle woven in parenthetically
   (e.g. "(Experiment B)"), at most one colon, em-dash only in the answer half.
2. **The question** — 1–3 sentences: the hypothesis and why we're testing it.
3. **Setup / experimental details** — a **compact table for parameters** (model, data, N, steps,
   lr/β, seeds). Tables are for *details*, not results.
4. **Results** — **point to the figure and describe the trend.** Inline only the few load-bearing
   numbers. Push the full per-condition numbers into the figure (ideally annotated) or the results doc.
5. **Conclusions** — **bullets, one claim each**, with a **bold lead**. Never a dense multi-claim
   paragraph.
6. **Caveats** — bullets.
7. **Artifacts** — one line: scripts, run-name globs, dataset paths, and the results-doc link.

## Rules

- **Prefer graphs to tables for results.** If a figure already shows the result, cite it and
  describe the trend — don't restate it as a table. Reserve tables for experimental details or for
  small categorical comparisons no figure captures.
- **Annotate the graph** with the key number/trend; the figure is the preferred home for numbers.
  (This may mean editing the plot script — worth it.)
- **Bullets over dense paragraphs.** Any ≥3-claim paragraph becomes a list.
- **Notation / definitions → a bullet list**, not a running paragraph.
- **Low number-density in prose.** Inline the headline contrast only (e.g. "46% vs DPO's 53%");
  the full sweep lives in the figure / results doc behind a pointer.
- **One idea per sentence; bold the lead** of each bullet so it's skimmable.
- **Cross-reference liberally** — link findings (#N), results docs, figures, and code by name.
- **Math** — piecewise in brace/`cases` form; keep formulas in a table cell or a notation bullet.

## Worked judgements on the current doc

**Keep:**
- ✅ #25's ladder table (`loss | gradient | …`) — a categorical comparison no figure captures.
- ✅ #23's table — but frame it as *experimental details*, not the result.
- ✅ #21's bold-lead bullets; #16's overall section shape.

**Fix:**
- ❌ #25 "Per-lr late-mean elicit (3 seeds; …)" — a number dump. Replace with a pointer to
  `signed_linear_lr_sweep.png` plus the single comparison that matters.
- ❌ #25 "Three firm conclusions" packed into one paragraph → three bullets.
- ❌ #25 notation as a running paragraph → bullet list.
- ❌ #26 results table (rank | arm 1 | arm 2) → drop; `swap_rank_sweep.png` already shows it, so
  describe the trend instead.
- ⚠️ #16 — well-structured but number-dense; thin the inline numbers and lean on its figure.
