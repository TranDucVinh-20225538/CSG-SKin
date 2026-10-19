# Paper B R6 R1-hi — high-λ shift 0% (RECIPE=current, last.ckpt)

Design: `decisions/paperB_r6_r1_hi.md`. Commit `7695913`.
**On origin/paperb/reviewer-r1:** _(fill after push)._

Degraded: ID bal < 0.25 (flagged, excluded). Gate: held-out MLP ≤ 0.55 (P6 V1).
Predictions apply to gated, non-degraded seeds only.

Fallback status: λ=8 gate met; adv-LR×3 not run.

## Cells

| wave | λ | n | degraded | gate pass | ID bal | ΔID vs R1 shift0 λ2 | primary no-inv | opposite inv |
|---|---:|---:|---:|---:|---|---|---|---|
| main | 4.0 | 5 | 0 | 0/5 | 0.483 ± 0.016 | 0.069 | None | None |
| main | 8.0 | 5 | 0 | 1/5 | 0.467 ± 0.021 | 0.053 | True | False |

### wave=main λ=4.0

| seed | ID bal | Δ vs R1 λ2 | deg | heldout lin/MLP/kNN | pad_full lin | gate | Maha Fitz [CI] | Maha MED [CI] | Maha SD [CI] |
|---:|---|---|---|---|---|---|---|---|---|
| 42 | 0.466 | -0.013 | False | 0.640/0.660/0.691 | 0.778 | False | 0.683 [0.669,0.696] | 0.530 [0.486,0.575] | 0.685 [0.672,0.698] |
| 52 | 0.479 | +0.340 | False | 0.715/0.732/0.711 | 0.808 | False | 0.650 [0.636,0.664] | 0.560 [0.518,0.606] | 0.668 [0.656,0.681] |
| 62 | 0.491 | +0.015 | False | 0.620/0.676/0.679 | 0.781 | False | 0.629 [0.614,0.644] | 0.507 [0.467,0.549] | 0.635 [0.621,0.648] |
| 72 | 0.473 | -0.007 | False | 0.668/0.664/0.727 | 0.803 | False | 0.733 [0.720,0.746] | 0.567 [0.531,0.609] | 0.725 [0.712,0.737] |
| 82 | 0.507 | +0.009 | False | 0.607/0.598/0.661 | 0.761 | False | 0.636 [0.622,0.650] | 0.513 [0.469,0.557] | 0.623 [0.609,0.636] |

### wave=main λ=8.0

| seed | ID bal | Δ vs R1 λ2 | deg | heldout lin/MLP/kNN | pad_full lin | gate | Maha Fitz [CI] | Maha MED [CI] | Maha SD [CI] |
|---:|---|---|---|---|---|---|---|---|---|
| 42 | 0.501 | +0.022 | False | 0.646/0.696/0.710 | 0.801 | False | 0.713 [0.700,0.726] | 0.544 [0.501,0.591] | 0.722 [0.710,0.734] |
| 52 | 0.460 | +0.322 | False | 0.617/0.545/0.651 | 0.774 | True | 0.730 [0.718,0.744] | 0.484 [0.439,0.528] | 0.716 [0.704,0.729] |
| 62 | 0.475 | -0.002 | False | 0.651/0.620/0.671 | 0.786 | False | 0.708 [0.694,0.721] | 0.539 [0.499,0.582] | 0.720 [0.708,0.732] |
| 72 | 0.451 | -0.028 | False | 0.603/0.634/0.667 | 0.753 | False | 0.731 [0.718,0.744] | 0.528 [0.491,0.566] | 0.735 [0.722,0.746] |
| 82 | 0.448 | -0.050 | False | 0.598/0.590/0.643 | 0.760 | False | 0.650 [0.636,0.664] | 0.486 [0.450,0.524] | 0.644 [0.630,0.657] |
Gated seeds: [52]. Non-inverting sets (≥ CI lo>0.5): ['fitzpatrick17k', 'sd198']; inverting sets (CI hi<0.5): [].

## Detectors

All five detectors with cluster CIs are in `per_run.json` → `detectors`.
Primary prediction / opposite reading use Mahalanobis on Fitz / MED-NODE / SD-198 only.

