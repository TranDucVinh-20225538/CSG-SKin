# Paper B R7 C2: figure and clinical run-set audit

Design: `decisions/paperB_r7_c2.md`. Commit `b0e93c1`.

`FIGURES_REPORT.md` is stale. Audit is against `scripts/plot_manuscript_figures.py`.

| Panel | Script | Run set | Checkpoint | Note |
|---|---|---|---|---|
| Fig 1a–d | `fig1_intervention` | Table 1 last.ckpt (r5/t1last/summary.csv) | last.ckpt | load_t1last_by_lambda; OOD ECE from UNSEEN_ECE.json |
| Fig 2 | `fig2_confidence` | phase13/per_run (STALE vs Table 1) | phase13 (best-era) | still load_phase13_runs; regenerate from t1last below |
| Fig 3 | `fig3_detectors` | phase3_sweep summaries | phase3 | load_phase3_summaries |
| Fig 4–6 / S* | `(various)` | see plot_manuscript_figures.py | mixed / legacy | FIGURES_REPORT.md header is stale; trust the script |
| §4.4 / App clinical % | `clinical_t1last` | Table 1 last.ckpt (clinical_t1last) | last.ckpt | /data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/r5/posthoc/clinical_t1last/results.json |

## Fig 2: phase13 (old) vs t1last last.ckpt (new)

Fig 2 still calls `load_phase13_runs()`. Regenerated MSP confidence from `results/paperB/r5/t1last/features_last` (Table 1 last.ckpt).

| λ | set | old mean | new mean | Δ |
|---:|---|---:|---:|---:|
| 0.0 | isic_test | 0.915 | 0.892 | -0.022 |
| 0.0 | pad_heldout | 0.489 | 0.444 | -0.046 |
| 0.0 | fitzpatrick17k | 0.601 | 0.665 | +0.065 |
| 0.25 | isic_test | 0.910 | 0.892 | -0.018 |
| 0.25 | pad_heldout | 0.898 | 0.909 | +0.011 |
| 0.25 | fitzpatrick17k | 0.909 | 0.921 | +0.012 |
| 0.5 | isic_test | 0.908 | 0.892 | -0.016 |
| 0.5 | pad_heldout | 0.911 | 0.924 | +0.013 |
| 0.5 | fitzpatrick17k | 0.914 | 0.935 | +0.022 |
| 1.0 | isic_test | 0.909 | 0.894 | -0.015 |
| 1.0 | pad_heldout | 0.923 | 0.936 | +0.013 |
| 1.0 | fitzpatrick17k | 0.925 | 0.923 | -0.002 |
| 2.0 | isic_test | 0.910 | 0.893 | -0.017 |
| 2.0 | pad_heldout | 0.930 | 0.936 | +0.006 |
| 2.0 | fitzpatrick17k | 0.935 | 0.927 | -0.009 |
| 4.0 | isic_test | 0.908 | 0.887 | -0.021 |
| 4.0 | pad_heldout | 0.916 | 0.927 | +0.011 |
| 4.0 | fitzpatrick17k | 0.924 | 0.937 | +0.013 |
| 8.0 | isic_test | 0.900 | 0.884 | -0.016 |
| 8.0 | pad_heldout | 0.906 | 0.918 | +0.013 |
| 8.0 | fitzpatrick17k | 0.901 | 0.923 | +0.022 |

Clinical percentages in §4.4 / App. H: `clinical_t1last` (Table 1 last.ckpt), already aligned.
No `.tex` edits in this item.

