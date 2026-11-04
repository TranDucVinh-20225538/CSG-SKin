# Paper B R10 consolidated report

GPU hours (G1–G3): **15.53**. C5 fixed (`z`→`z_lesion_norm`). Follow-up reports regenerated with `--no_boot` where noted in `decisions/paperB_r10.md` Deviations.

## C0 degraded audit

See `DEGRADED_AUDIT.md`. Class-aware shift0% λ=2 after excluding s52: (a) 0.630, (b) 0.522, (c) 0.555.

## G1 — shift 50% (5 seeds)

| λ | ID | Leak | pad | Fitz | MED | SD-198 | SCIN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.484 ± 0.008 | 0.637 ± 0.039 | 0.486 ± 0.047 | 0.455 ± 0.115 | 0.312 ± 0.051 | 0.428 ± 0.104 | 0.567 ± 0.080 |
| 2.0 | 0.487 ± 0.009 | 0.647 ± 0.021 | 0.469 ± 0.027 | 0.457 ± 0.062 | 0.322 ± 0.054 | 0.432 ± 0.047 | 0.583 ± 0.043 |

Old → new (λ=2 pad, 3→5 seed): 0.454 ± 0.009 → 0.469 ± 0.027.

Leakage-matched (n=79 runs): verdict (a) **supported**, (b) **supported**. Shift coef Fitz −0.167, MED −0.108, SD −0.173 (CIs exclude 0).

Class-aware (G1 cells):
| cell | n | (a) | (b) | (c) |
|---|---:|---:|---:|---:|
| R1 shift50% λ=0.5 | 5 | 0.527 ± 0.028 | 0.528 ± 0.025 | 0.581 ± 0.016 |
| R1 shift50% λ=2 | 5 | 0.544 ± 0.011 | 0.543 ± 0.016 | 0.589 ± 0.007 |

Clinical post hoc includes R1 shift 50% λ=2 (`r5/posthoc/clinical/REPORT.md`).

## G2 — pad_adv 50% λ=0

| arm | λ | Leak | ID | pad | Fitz |
|---|---:|---:|---:|---:|---:|
| pad_adv 25% | 0 | 0.898 ± 0.032 | 0.472 ± 0.019 | 0.819 ± 0.022 | 0.724 ± 0.043 |
| pad_adv 25% | 2 | 0.632 ± 0.007 | 0.477 ± 0.002 | 0.479 ± 0.013 | 0.407 ± 0.011 |
| pad_adv 50% | 0 | 0.912 ± 0.030 | 0.482 ± 0.010 | 0.832 ± 0.010 | 0.713 ± 0.026 |
| pad_adv 50% | 2 | 0.575 ± 0.030 | 0.459 ± 0.022 | 0.415 ± 0.043 | 0.397 ± 0.069 |

New row 50% λ=0: leak 0.912±0.030, pad 0.832±0.010 (T2 REPORT regenerated).

## G3 — IWDAN n50 λ=0.5

TV 0.118 ± 0.017; pad 0.665 ± 0.060; Fitz 0.741 ± 0.036; (a) 0.647 ± 0.049; (b) 0.560 ± 0.042; (c) 0.584 ± 0.023. Not Rule A.

See `G3_n50_l05.md`.

## C1 — within-class (c) at λ=0.5 for tab:prior

| arm | n | (a) | (b) | (c) |
|---|---:|---:|---:|---:|
| R1 shift100% λ=0.5 | 5 | 0.510 ± 0.013 | 0.604 ± 0.070 | 0.618 ± 0.010 |
| R2 IWDAN λ=0.5 | 5 | 0.548 ± 0.031 | 0.570 ± 0.018 | 0.588 ± 0.006 |
| R2 IWDAN_ORACLE λ=0.5 | 5 | 0.617 ± 0.038 | 0.533 ± 0.028 | 0.562 ± 0.015 |
| R2-prior n50 λ=0.5 | 5 | 0.647 ± 0.049 | 0.560 ± 0.042 | 0.584 ± 0.023 |

## C2 — tab:dial extras (λ=2 headline)

| set | Maha | Rule A |
|---|---:|---|
| mednode | 0.288±0.027 | met |
| sd198 | 0.376±0.026 | met |
| scin | 0.551±0.039 | not met |

## C3 — tab:unseen DINOv2 / A4

| bb | λ | pad | Fitz | MED | SD | SCIN |
|---|---:|---:|---:|---:|---:|---:|
| dinov2 | 0.0 | 0.831±0.029 | 0.778±0.016 | 0.580±0.030 | 0.790±0.005 | 0.716±0.041 |
| dinov2 | 2.0 | 0.448±0.045 | 0.472±0.067 | 0.363±0.045 | 0.461±0.066 | 0.551±0.064 |
| a4 | 0.0 | 0.786±0.019 | 0.821±0.023 | 0.560±0.021 | 0.799±0.026 | 0.822±0.022 |
| a4 | 2.0 | 0.452±0.014 | 0.458±0.061 | 0.298±0.038 | 0.430±0.061 | 0.659±0.073 |

## C4 — SCIN in shift / sixclass / splits

- tab:shift: R1 REPORT SCIN dose — shift 50% λ=2 Maha **0.583±0.043**; λ=0.5 **0.567±0.080**.
- tab:sixclass: R9 E1 REPORT regenerated (SCIN in ALL_OODS).
- R2 SCIN: `r6/r2/cur/REPORT_SCIN.md` regenerated (`--no_boot`).

## C5 — tab:single-dino

MED 0.348 ± 0.047; SD 0.489 ± 0.065; SCIN 0.539 ± 0.056; pad 0.546 ± 0.033.

## C6

Fig. 1b already from clinical CSV (prior commit).

## Paper numbers changed (old → new)

- tab:shift 50% λ=2 pad (3→5 seed): 0.454 ± 0.009 → 0.469 ± 0.027
- tab:shift 50% λ=2 Fitz: 0.441 ± 0.076 → 0.457 ± 0.062
- tab:padadv: **new** pad_adv 50% λ=0 leak 0.912±0.030 / pad 0.832±0.010
- tab:prior: **new** n50 λ=0.5 pad 0.665±0.060 / TV 0.118±0.017 / (c) 0.584±0.023
- leakage-matched pool: n=79 (includes shift50×5×2λ)
- class-aware: new λ=0.5 / shift50 / n50 cells

## GPU detail

| job | hours |
|---|---:|
| 72282_0 paperB-r10-g1 | 1.517 |
| 72282_1 paperB-r10-g1 | 1.536 |
| 72282_2 paperB-r10-g1 | 1.55 |
| 72282_3 paperB-r10-g1 | 1.533 |
| 72283_0 paperB-r10-g2 | 1.166 |
| 72283_1 paperB-r10-g2 | 1.164 |
| 72283_2 paperB-r10-g2 | 1.168 |
| 72284_0 paperB-r10-g3 | 1.181 |
| 72284_1 paperB-r10-g3 | 1.188 |
| 72284_2 paperB-r10-g3 | 1.18 |
| 72284_3 paperB-r10-g3 | 1.161 |
| 72284_4 paperB-r10-g3 | 1.189 |
| **total** | **15.53** |
