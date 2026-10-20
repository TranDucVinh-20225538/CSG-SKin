# (7) ERM reference and R2 IWCDAN / CDAN / oracle / λ = 0.5

## ERM single-encoder reference (lesion-level, λ = 0)

Path: `r5/p8/erm_ref/single_dann_ladv0_s*`. **Available now: n = 5** of 5 — complete (5/5).

ID bal acc: **0.558 ± 0.022** (SD); **SE = 0.010** (SD/√n). Per-seed: 0.567, 0.578, 0.535, 0.535, 0.578.

## R2 cells (recipe `current`, last.ckpt) — point mean ± SD; CIs in `r6/r2/cur/REPORT.md`

| method | λ | n | Maha pad_heldout | Maha Fitz | ID bal | leakage |
|---|---:|---:|---|---|---|---|
| CDAN | 0.5 | 5 | 0.372 ± 0.019 | 0.403 ± 0.039 | 0.474 ± 0.011 | 0.581 ± 0.044 |
| CDAN | 2 | 5 | 0.378 ± 0.033 | 0.420 ± 0.052 | 0.460 ± 0.015 | 0.605 ± 0.038 |
| IWCDAN | 0.5 | 5 | 0.389 ± 0.024 | 0.410 ± 0.032 | 0.424 ± 0.014 | 0.561 ± 0.019 |
| IWCDAN | 2 | 5 | 0.360 ± 0.034 | 0.384 ± 0.046 | 0.450 ± 0.014 | 0.641 ± 0.058 |
| IWDAN | 0.5 | 5 | 0.667 ± 0.026 | 0.633 ± 0.058 | 0.482 ± 0.021 | 0.665 ± 0.034 |
| IWDAN | 2 | 5 | 0.513 ± 0.014 | 0.508 ± 0.078 | 0.484 ± 0.013 | 0.642 ± 0.042 |
| IWDAN_ORACLE | 0.5 | 5 | 0.680 ± 0.026 | 0.742 ± 0.027 | 0.395 ± 0.008 | 0.760 ± 0.022 |
| IWDAN_ORACLE | 2 | 5 | 0.647 ± 0.032 | 0.717 ± 0.044 | 0.398 ± 0.002 | 0.780 ± 0.022 |

Note: CDAN / IWDAN / IWCDAN / IWDAN_ORACLE at λ ∈ {0.5, 2} are **5/5**. Full cluster CIs + unseen MED/SD + class-aware probes: `results/paperB/r6/r2/cur/REPORT.md`.
