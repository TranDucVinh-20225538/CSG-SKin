# (7) ERM reference and R2 IWCDAN / CDAN / λ = 0.5

## ERM single-encoder reference (lesion-level, λ = 0)

Path: `r5/p8/erm_ref/single_dann_ladv0_s*`. **Available now: n = 2** of 5 (remaining ERM seeds still queued/held on GPU).

ID bal acc: **0.572 ± 0.008** (SD); **SE = 0.005** (SD/√n). Per-seed: 0.567, 0.578.

## R2 cells (recipe `current`, last.ckpt) — point mean ± SD over available seeds

| method | λ | n | Maha pad_heldout | Maha Fitz | ID bal | leakage |
|---|---:|---:|---|---|---|---|
| CDAN | 0.5 | 2 | 0.377 ± 0.005 | 0.391 ± 0.038 | 0.474 ± 0.013 | 0.602 ± 0.011 |
| CDAN | 2 | 5 | 0.378 ± 0.033 | 0.420 ± 0.052 | 0.460 ± 0.015 | 0.605 ± 0.038 |
| IWCDAN | 0.5 | 5 | 0.389 ± 0.024 | 0.410 ± 0.032 | 0.424 ± 0.014 | 0.561 ± 0.019 |
| IWCDAN | 2 | 5 | 0.360 ± 0.034 | 0.384 ± 0.046 | 0.450 ± 0.014 | 0.641 ± 0.058 |
| IWDAN | 0.5 | 5 | 0.667 ± 0.026 | 0.633 ± 0.058 | 0.482 ± 0.021 | 0.665 ± 0.034 |
| IWDAN | 2 | 5 | 0.513 ± 0.014 | 0.508 ± 0.078 | 0.484 ± 0.013 | 0.642 ± 0.042 |

Note: CDAN λ = 0.5 currently **2/5** seeds (42, 52); others complete at 5/5. Oracle λ = 0.5 not started. Cluster CIs for incomplete cells deferred until n = 5.
