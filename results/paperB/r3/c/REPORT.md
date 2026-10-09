# R3 C — Figures 1, 2, 3, 5 from the lesion-level sweep

Code: `scripts/r3_c_lesion_figures.py` → `scripts/plot_manuscript_figures.py` (unchanged plotting, palette #0072B2/#D55E00/#009E73/#000000/#E69F00, constrained_layout, 300 dpi, PDF + PNG, grayscale PNG). Figure 6 untouched. Image-level versions kept as `<stem>_imagelevel.{pdf,png}`.

**Verdict:** All seven λ plotted in every panel.

Fig 1 shared y-range (a)(b)(c): 0.35–1.00, the smallest 0.05 step that contains every mean ± s.d. (lowest 0.366).

| λ | seeds (summary.json: Fig 1a–c, 3) | seeds (per-run analysis: Fig 1d, 2, 5) |
|---:|---|---|
| 0 | 42, 52, 62, 72, 82 | 42, 52, 62, 72, 82 |
| 0.25 | 42, 43, 44 | 42, 43, 44 |
| 0.5 | 42, 43, 44 | 42, 43, 44 |
| 1 | 42, 43, 44 | 42, 43, 44 |
| 2 | 42, 52, 62, 72, 82 | 42, 52, 62, 72, 82 |
| 4 | 42, 43, 44 | 42, 43, 44 |
| 8 | 42, 43, 44 | 42, 43, 44 |

## Missing

- None.

## Caveats

- Fig 1c is Mahalanobis on pad_heldout from the lesion-level summaries; the image-level Fig 1c used the Phase 16 dual-column file.
- λ ∈ {0, 2} use seeds 42/52/62 (+72/82 once B1 lands); other λ use 42/43/44.
