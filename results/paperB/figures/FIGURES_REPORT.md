# Manuscript figures report

Outputs: `results/paperB/figures/` at 300 dpi PDF+PNG.

## Data sources
- Fig 1a–c: `phase3_sweep/*/summary.json`
- Fig 1d, 2, 5: `phase13/per_run/*.json` (`confidence`, `depths_pad_full`)
- Fig 1c, 3, 4: PAD held-out / Fitz from Phase 3 summaries
- Fig 4 refs: `phase2_5/phase25_aggregate.json` (Fitz baselines)
- Fig 6a: `phase16/b1/dense3seed_aggregate.json` (λ ∈ {0, 0.25, 1} only)
- Fig 6b: `phase16/b1_rev` dense3seed + `table1_ood_pad_dual_column.json`
- S1: `phase4_semantic_ood/phase4_aggregate.json`
- S2: `phase1_6/domain_axis_projections.npz`
- S3: Phase 1.5 preview points (`plot_preview_curve.py`)

## Notes
- λ axis is **categorical** (even spacing); see figure captions.
- Fig 6a uses three λ points (dense 3-seed protocol); not a full seven-λ grid.
- Fig S3 plots leakage on x-axis (supplementary, explicitly superseded); main figures do not.

## Grayscale exports
# Grayscale distinguishability (line/marker)
- fig1_intervention.png: saved fig1_intervention_gray.png
- fig2_confidence.png: saved fig2_confidence_gray.png
- fig3_detectors.png: saved fig3_detectors_gray.png
- fig4_generalises.png: saved fig4_generalises_gray.png
- fig5_leakage_no_mediation.png: saved fig5_leakage_no_mediation_gray.png
- fig6_site_shift.png: saved fig6_site_shift_gray.png
- figS1_semantic_ood.png: saved figS1_semantic_ood_gray.png
- figS2_domain_axis.png: saved figS2_domain_axis_gray.png
- figS3_preview_superseded.png: saved figS3_preview_superseded_gray.png
