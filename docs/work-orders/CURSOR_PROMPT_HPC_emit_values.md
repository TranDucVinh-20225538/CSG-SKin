# TASK — emit the remaining manuscript values

Small extraction job, no training, no new analysis. Write
`scripts/emit_tex_values.py` and run it. It must print a block of LaTeX
`\newcommand` lines that paste directly into the FILL-IN BLOCK at the top of
`paper_b.tex`.

## Values needed

### 1. Cross-domain balanced accuracy, remaining $\lambda$ points

From `results/paperB/phase6_xfer/` (the 37 JSONs). Already in the manuscript:
$\lambda = 0$: `0.291 ± 0.016`; $\lambda = 0.25$: `0.291 ± 0.020`;
$\lambda = 2$: `0.249 ± 0.030`.

Emit the same statistic — **six-class balanced accuracy on `pad_heldout`,
mean ± s.d. across seeds** — for $\lambda \in \{0.5, 1, 4, 8\}$.

If any of these were never computed, say so explicitly and emit `n/a` rather
than substituting a different metric or a different PAD subset.

### 2. EffNet-B3 single-encoder control, cross-domain balanced accuracy

Same metric, same `pad_heldout` split, for the EffB3 single-encoder control.
The manuscript currently cites the ResNet-50 baseline at `0.391`; the EffB3
control is the primary control and its row is empty.

**Verify the split before reporting**: it must be `pad_heldout` (716 images,
412 patients), not `pad_full` or `pad_adv`. If the control was only ever
evaluated on a different subset, report that fact rather than the number.

### 3. Per-class counts in `pad_heldout`

Counts of **AK** and **SCC** in the 716-image patient-level held-out split.
Melanoma is already recorded as $n = 9$.

These gate a claim in the Discussion. For reference, a proportional 31.2%
split of full PAD would give AK $\approx 227$ and SCC $\approx 60$, but
patient-level splitting does not preserve class proportions exactly — melanoma
came out at 9 against a proportional 16 — so report the **actual** counts.

Also emit, for the report only (not the manuscript): the per-class recall at
$\lambda = 0$ and $\lambda = 2$ with its raw numerator and denominator, e.g.
`AK 56/227 → 6/227`. If AK's denominator is below ~100, flag it — the
manuscript's strongest Discussion sentence rests on it.

## Output format

Print exactly this, with values substituted:

```latex
\newcommand{\xdomHalf}{0.xxx\,$\pm$\,0.xxx}
\newcommand{\xdomOne}{0.xxx\,$\pm$\,0.xxx}
\newcommand{\xdomFour}{0.xxx\,$\pm$\,0.xxx}
\newcommand{\xdomEight}{0.xxx\,$\pm$\,0.xxx}
\newcommand{\xdomEffb}{0.xxx\,$\pm$\,0.xxx}
\newcommand{\nAK}{xxx}
\newcommand{\nSCC}{xxx}
```

Three decimal places, matching the rest of Table 1. Write the block to
`results/paperB/TEX_VALUES.md` alongside a short note stating, for each value,
which file it came from and which PAD subset it used.

## Rules

Report `n/a` with an explanation rather than a substituted or approximated
number. Do not recompute anything with a different split, metric or class
restriction than the ones already in the manuscript — a silently different
denominator here would be worse than an empty cell.
