# Historical Fitzpatrick / recovery anchors (verified pre-report)

Source: `results/paperB/r6/r1/cur/REPORT.md` (resampling protocol; cross-protocol vs R12 reweight).

## Locked AUROC means (Mahalanobis, λ=2, seed mean±sd)

| series | Fitzpatrick17k | MED-NODE | SD-198 | pad_heldout |
|---|---:|---:|---:|---:|
| R1 shift 100% | **0.340 ± 0.065** | 0.298 ± 0.032 | 0.325 ± 0.069 | 0.368 ± 0.017 |
| R1 shift 50% | 0.457 ± 0.062 | 0.322 ± 0.054 | 0.432 ± 0.047 | 0.469 ± 0.027 |
| R1 shift 0% | **0.691 ± 0.031** | 0.541 ± 0.027 | 0.694 ± 0.031 | 0.458 ± 0.003 |
| R1 λ=0 baseline (shift100) | 0.712 ± 0.013 | — | — | 0.825 ± 0.022 |

Do **not** replace 0.340 by 0.405 (or vice versa) across series. The figure **0.405** appears in R1 stratified / other cells (e.g. shift0 λ2 ID-class blocks) — keep attached to its own table.

## Recovery % (R1 formula: (shiftX − shift100) / (shift0 − shift100) at λ=2)

| set | shift50 recovery | notes |
|---|---:|---|
| Fitzpatrick17k | (0.457−0.340)/(0.691−0.340) = **0.333** | not ~45% |
| MED-NODE | (0.322−0.298)/(0.541−0.298) = **0.099** | not ~20% |
| SD-198 | (0.432−0.325)/(0.694−0.325) = **0.290** | not ~43% |

Δ(shift0−shift100) from R1 table: Fitz +0.351, MED +0.243, SD +0.368 (these are absolute AUROC lifts, sometimes paraphrased as percentages of a different baseline in prose — verify before any corrected summary).

R12 REPORT must reconcile recovery only within matched experimental series (reweight vs reweight; R1 only as cross-protocol reference).
