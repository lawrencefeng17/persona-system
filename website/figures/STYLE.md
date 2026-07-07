# Blog figure style guide (`website/figures/`)

All blog figures are built by `build_blog_figs.py` (repo root) into this folder.
Every figure must follow these rules so the post reads as one visual system.

## Layout & output

- **High DPI, no wasted space:** save at `dpi=300` with `bbox_inches="tight"`,
  `pad_inches=0.05`, and `fig.tight_layout()`. Size the figure so the axes fill
  the canvas width under the legend (a legend wider than the axes creates dead
  side margins — widen the figure instead; fig 1 uses `figsize=(9.5, 4.5)`).
- **Large fonts.** Baseline rcParams:

  ```python
  plt.rcParams.update({
      "font.size": 14,
      "axes.labelsize": 16,
      "xtick.labelsize": 14,
      "ytick.labelsize": 14,
      "legend.fontsize": 13,
      "axes.titlesize": 16,
  })
  ```

- **Keep the full box border** (all four spines). Light dashed grid
  (`alpha=0.3, ls="--"`) behind the data.

## Legends

- **Never inside the axes.** Place above (preferred) or below the plot:
  `ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=..., frameon=False)`.
- **Handles must show the real mark.** Use the errorbar/line containers as
  handles so caps and line segments are visible; `handlelength >= 2` and no
  marker downscaling.
- **Descriptive labels:** spell out the condition — "LoRA, 250k examples",
  "full fine-tuning, 1M examples", "untrained baseline". Keep the exact same
  wording for the same condition across every figure.
- Reorder handles explicitly (`ax.get_legend_handles_labels()` + index list) —
  matplotlib lists plain lines (e.g. an `axhline` baseline) before errorbar
  containers, which puts the baseline first unless you override.

## Axes

- **Short labels** with units in parens: "stories mentioning owl (%)",
  "LoRA rank". No sentence-length axis labels.
- Trim horizontal padding with an explicit `set_xlim` close to the data.
- Percentage axes span the full 0–105 range when the story is
  distance-to-ceiling; don't zoom in to exaggerate differences.

## Color semantics (consistent across ALL blog figures)

A color means the same thing in every figure it appears in. Current registry:

| meaning | color | source |
|---|---|---|
| 10k examples | `#9ECAE1` (light blue) | data-scale gradient |
| 26k examples | `#4292C6` | data-scale gradient |
| 250k examples | `#1F77B4` (matplotlib default blue, C0) | data-scale gradient anchor |
| 500k examples | `#0B559F` | data-scale gradient |
| 1M examples | `#08306B` (dark navy) | data-scale gradient |
| best-lr-per-rank frontier + lift arrows | soft orange `#F28E2B`, markers filled the same, no outline | fixed; complement of the post's blue, kept unharsh |
| shared-lr (untuned) curve | C0 blue `#1F77B4` (the post's blue family) | fixed |
| annotations | dark gray `#333333` | fixed |
| untrained baseline | gray, dotted (`ls=":"`, `#808080`) | fixed |
| learning rate as a curve family (appendix B) | Oranges ramp, darker = larger rate | extends fig 2's orange = learning-rate semantics |
| LoRA rank as a curve family (appendix C) | Greens ramp, darker = higher rank; a colorbar may stand in for the legend when a gradient has many levels | blue stays reserved for data scale |
| full fine-tuning as its own curve (appendix C) | dark gray `#444444`, diamond markers | shape still encodes method |
| teacher-forced probability (appendix E) | C0 blue `#1F77B4` | continuous measure |
| sampled elicitation rate on a trajectory (appendix E) | soft orange `#F28E2B` | discrete measure |
| shared-seed release vs fresh data (appendix G) | release = medium gray `#999999`, fresh = C0 blue | |
| DPO on number pairs (appendix H) | soft purple `#9467BD` (C4), square markers | new training objective = new hue + marker |

Markers are plain fills — no dark edge outlines.

Two condition encodings that are NOT color:
- **Same data repeated for more epochs** wears its dataset's color with a
  dashed line and star markers (appendix D) — epochs are not new data.
- **Data provenance** (released by prior work vs freshly generated) is marker
  FILL: open = released, filled = fresh (appendix C).

When the same condition appears in two panels (e.g. fig 2's shared-2e-4 curve,
drawn full-strength on the left and ghosted at low alpha on the right), it
wears the same color in both so the panels cross-reference.

On a best-per-rank frontier, prefer showing ONLY the frontier and encoding
the winning lr in the markers (value annotated beside each point) over
drawing every per-lr curve faintly — the full grid belongs in an appendix
heatmap, not under the headline line. Marker DIAMETER is proportional to the
lr itself (halving the lr halves the marker; fig 2 uses `28 * lr / 8e-4`,
floored at 2.8pt so the smallest winners stay findable). Size and the
annotation carry the rate — do NOT also encode it in hue; markers wear the
frontier color.

- **Ordered variables get gradients:** one hue family, darker = more (data
  scale uses the blue ramp above; extend it toward lighter blues for 10k/26k
  if those scales appear). Never assign unrelated hues to the levels of an
  ordered variable.
- **Color encodes data, not method.** Method (LoRA vs full fine-tuning) is
  encoded by **marker shape** — circle `o` for LoRA, diamond `D` for FFT — plus
  the vertical separator bar and its own x-tick. So the FFT point at 250k wears
  the *same* color as the LoRA-at-250k curve.
- Primary color family for this post is **blue, anchored at the matplotlib
  default C0**. Reserve other hues for genuinely new meanings and add them to
  the registry above when introduced.

## Error bars

- Always SEM over seeds, `capsize=5`, same color as the mark.
- State n in the caption, not on the plot. If the bars are smaller than the
  markers (common at ceiling), say so in the caption rather than inflating
  marker-free styling.
