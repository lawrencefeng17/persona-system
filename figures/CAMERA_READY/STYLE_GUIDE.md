# Camera-ready figure style guide

Rules for every figure that ships in the blog post. Cleaned figures live in this folder
(`figures/CAMERA_READY/`); the originals stay where they are for the internal findings docs.

## Color
- Default matplotlib colors are fine for categorical series.
- **Reserve the red→green spectrum (e.g. `RdYlGn`) for TEST PERFORMANCE only** (elicitation /
  leakage rate). Do not use a color spectrum to encode anything else.
- **Rank is encoded by marker size** (see `memorization_map` for the pattern). When a
  performance colormap (red→green) is present in the same figure, use size ONLY for rank.
- When NO performance colormap is in the figure, rank may ALSO use a color spectrum, but it
  must be a spectrum DISTINCT from red→green (use `viridis`) so the two are never confused.
  `loss_vs_transfer` uses size + viridis for rank; `data_scaling_overlay` uses viridis.

## Titles and captions
- **Avoid titles.** No multi-line titles. Prefer a substantive caption written in the blog
  markdown, ML-conference style (the caption carries the explanation, not a plot title).
- If a one-line title is unavoidable, keep it short and plain.

## Axes, legends, labels
- Plain English everywhere a reader sees it. **No internal jargon**: drop `x26`, `late-window`,
  `omit_system`, `best-of-LR`, `coherence-gated`, `LLS`, finding numbers (`#37`, `§18`), and any
  reference to failed/abandoned experiments.
- Name the actual setting: dataset size, trait, and what each axis means.
  - e.g. colorbar `elicit: cat (%)` → `Rate of picking animal when asked`.
  - e.g. legend `coherence-gated` → `Best 100% coherent checkpoint at every rank`.
- Keep legends simple. Drop complex multi-entry legends and tiny per-point annotations.

## Error bars
- Show error bars when possible.
- **State in the caption what the variance is over** (seeds, or test samples).

## Output
- Save cleaned figures into `figures/CAMERA_READY/`.
- Camera-ready builder scripts are prefixed `cr_` in the repo root.
