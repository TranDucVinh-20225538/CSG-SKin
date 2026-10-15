# Paper B R5 / P11 — moment matching instead of an adversary, compared with GRL at matched invariance

Committed before any P11 job runs. Fixed now; not changed after results are seen. Deviations are appended at the end.
This item absorbs Q2's CORAL arm (user decision); GroupDRO is dropped.

## Question

Is the inversion on pad_heldout specific to adversarial training, or does any objective that aligns the ISIC and PAD
marginals of `z_lesion_norm` to the same degree produce it? Earlier moment-matching attempts do not answer this:
the Phase 15 MMD (single EfficientNet-B3, image-level, fixed σ ∈ {1, …, 16} on 1,536-d features) reached leakage
≥ 0.836 and degraded the classifier, and the Phase 15b CORAL term (‖C_s − C_t‖²_F / 4d², no mean term) let the domain
means drift apart (CORAL anomaly). P11 uses properly scaled versions on the dual encoder's 16-d scoring space, tunes
them to the GRL runs' invariance (target leakage ≈ 0.60) with the classifier intact, and compares OOD behaviour at
matched leakage.

## Design (scripts/train_r5_dual_isic.py, `--penalty`)

Dual encoder EfficientNet-B3 (the Table 1 model), lesion-level split, λ_adv = 0 (adversary term off; context and
orthogonality terms unchanged), plus w · penalty(z_lesion_norm[ISIC rows], z_lesion_norm[PAD rows]) per batch
(32 + 32, domain-balanced):

- MK-MMD (Long et al. 2015): Gaussian kernels with bandwidths = median pairwise squared distance of the pooled batch
  (detached) × {0.25, 0.5, 1, 2, 4}; unbiased MMD² estimator; kernels averaged. Scale-free in the feature dimension.
- Scaled CORAL: (‖μ_s − μ_t‖² + ‖C_s − C_t‖²_F) / d, d = 16. The mean term closes the Phase 15b anomaly.

Stages (seed 42 first, all selection on validation-side numbers):

1. Control: no penalty, λ_adv = 0, seed 42 (the selection reference in this pipeline).
2. Scan: {MK-MMD, CORAL} × w ∈ {1, 3, 10, 30} at the recipe's epochs (40); longer schedule: {MK-MMD, CORAL} × w = 10 at
   80 epochs.
3. Selection (per penalty; `r5_p11_report.py --select`): among scan configs with "classifier intact" (ISIC val balanced
   accuracy ≥ 0.95 × the control's), the one whose selection probe (logistic probe ISIC val vs pad_adv on
   z_lesion_norm, training-side images only) is closest to 0.60; ties within 0.01 → smaller weight, then 40 epochs. If
   no config is intact: the one with the highest val balanced accuracy among those with selection probe ≤ 0.75,
   flagged "classifier not intact"; if none, the penalty is reported as "target not reached" and has no seed stage.
   No test, OOD or Table 1 number enters the selection.
4. Seeds: the selected config per penalty at seeds 52, 62, 72, 82 (seed 42 from the scan) → 5 seeds per penalty.

A single-encoder arm is not run here (P9 covers single encoders; the comparison of interest is with the dual-encoder
GRL runs of Table 1).

## Data / splits

Lesion-level ISIC split (train 16,211 / val 4,053 / test 5,067), PAD pad_adv (training) / pad_heldout (716 images / 412
patients), Fitzpatrick17k, P7a clinical sets when present.

## Seeds and grid

Scan: seed 42, w ∈ {1, 3, 10, 30} (40 epochs) and w = 10 (80 epochs), both penalties. Seeds stage: 42, 52, 62, 72, 82.

## Matched-invariance comparison with GRL

GRL comparators: the 30 published lesion-level GRL runs with λ > 0 (R2 item 4: λ ∈ {0.25, 0.5, 1, 2, 4, 8}, Table 1 /
R4). For each selected P11 cell with mean Table 1 leakage ℓ, the GRL runs whose leakage is within ±0.05 of ℓ form the
matched comparator (n, mean λ, pad_heldout and Fitzpatrick Mahalanobis, mean ± SD). Reported side by side; no test of
the difference is pre-registered (descriptive comparison, numbers only).

## OOD sets, detectors, metrics

Table 1 columns via `r5_common.eval_isic` (leakage = ISIC test vs PAD full probe; ID balanced accuracy; ID ECE; OOD ECE
pad_heldout; Mahalanobis pad_heldout; Fitzpatrick; X-dom six-class); kNN, cosine, MSP, Energy; NV-nearest fraction;
pad_heldout reweighted to the ISIC class mix (decomposition check); P7a sets.

## Gates (flag, never drop)

Classifier intact (above). Penalty value logged per epoch (adversary_log.jsonl `penalty`); a run whose penalty does not
fall below its epoch-1 value by the last epoch is flagged "penalty not optimised".

## Controls

The no-penalty λ_adv = 0 control (seed 42) and the published λ = 0 rows (5 seeds).

## CI / bootstrap and rule A

PREREGISTER_BOOTSTRAP.json unchanged; ID by lesion group, pad_heldout by patient, Fitzpatrick by image, SCIN by case.
Rule A (pooled CI below 0.5 and ≥ 3 of 5 seed CIs below 0.5, Mahalanobis) per selected cell on pad_heldout and
Fitzpatrick. The scan (one seed) has no CI.

## Reported cells

Scan table (penalty, weight, epochs, selection probe, val balanced accuracy, leakage, ID balanced accuracy, pad_heldout
Mahalanobis); selection; per selected penalty: Table 1 row (5 seeds), bootstrap rules, matched GRL comparator,
diagnostics; LaTeX rows; one plain paragraph per selected cell.

## Predicted outcome (written before any score exists)

- MK-MMD reaches a selection probe of 0.60–0.70 with the classifier intact at w = 10–30; scaled CORAL reaches
  0.65–0.75 (first and second moments leave higher-order differences a linear probe can still use).
- At matched leakage, pad_heldout Mahalanobis AUROC is below 0.5 for MK-MMD (0.40–0.48, rule A) and within 0.05 of the
  matched GRL runs; CORAL below 0.5 in mean (rule A or B). Fitzpatrick follows pad_heldout.
- Reading: the inversion is a property of aligning the two marginals under label shift, not of the adversary.

## Outputs

`results/paperB/r5/p11/runs/<tag>/` (config.json, adversary_log.jsonl, summary.json, scores.npz, features/);
`SELECTION.json`, `summary.csv`, `BOOTSTRAP.json`, `REPORT.md`. Checkpoints `checkpoints/r5_p11/`. `--recipe p8` runs
carry `_recp8` and report to `p11/recipe_recp8/`.

## GPU-h estimate and basis

Basis: lesion-level dual EfficientNet-B3 run ≈ 1.6 GPU-h at 40 epochs (3.2 at 80); the penalties act on 16-d vectors of
32 + 32 rows (negligible).

| Stage | Runs | GPU-h |
|---|---:|---:|
| Control | 1 | 1.6 |
| Scan, 40 epochs | 8 | 12.8 |
| Scan, 80 epochs | 2 | 6.4 |
| Seeds (2 penalties × 4) | 8 | 12.8 (both 40-epoch) to 25.6 (both 80-epoch) |
| Total | 19 | 33.6–46.4 |

The approved budget is ≈ 40 GPU-h; if both selections are 80-epoch configs the seed stage exceeds it by ≈ 6 GPU-h and
the user decides before `launch_r5_p11.sh seeds`. CPU: selection ≈ 0.05 h, report ≈ 0.3 h.

## Stop rules and caps

Scan fixed (11 runs). Seed stage only for a selected penalty. A failed run is re-run once with the same seed; a second
failure is reported.

## Deviations
