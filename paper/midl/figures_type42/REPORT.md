# Type-42 figure re-export (Paper B)

Old files under `paper/midl/figures/` and `results/paperB/figures/` were **not** modified.
New PDFs: `paper/midl/figures_type42/`.
Scripts: `paper/midl/figures/fig_mechanism.py` (`pdf.fonttype=42`), `scripts/paper_figure_style.py` (already had 42; `PAPERB_FIG_OUT` override).

## pdffonts

All new PDFs: **0 Type 3**; DejaVu embedded as **CID TrueType** (Type 42).

## 300 dpi diff vs old

| figure | old Type3 | new Type3 | mean ‖Δ‖ | frac>30 | verdict |
|---|---:|---:|---:|---:|---|
| fig_mechanism | 2 | 0 | 0.402 | 0.0043 | **REPLACE_OK** |
| fig1_intervention | 0 | 0 | 0.000 | 0.0000 | **REPLACE_OK** |
| fig2_confidence | 0 | 0 | 13.405 | 0.1651 | DO_NOT_REPLACE |
| fig3_detectors | 0 | 0 | 11.004 | 0.1389 | DO_NOT_REPLACE |
| fig4_generalises | 0 | 0 | 0.480 | 0.0072 | **REPLACE_OK** |
| fig5_leakage_no_mediation | 0 | 0 | 4.357 | 0.0669 | **REPLACE_OK** |
| fig6_site_shift | 0 | 0 | 0.000 | 0.0000 | **REPLACE_OK** |

## Can replace in manuscript

- `fig_mechanism.pdf` (was the only Type-3 figure in current `paper_b.tex`)
- `fig1_intervention.pdf` (already Type 42; bit-identical re-export)
- `fig4_generalises.pdf`, `fig5_leakage_no_mediation.pdf`, `fig6_site_shift.pdf` (appendix)

## Do not replace

- `fig2_confidence.pdf`, `fig3_detectors.pdf` — layout/bbox differs beyond font AA.
